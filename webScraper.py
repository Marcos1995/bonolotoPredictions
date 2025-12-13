"""
Web scraper for Bonoloto results from the official Spanish lottery website.
Uses Selenium to bypass anti-bot protection and extract latest raffle results.
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
import re
from datetime import datetime
import colorama
import commonFunctions as cf

# Try to import webdriver-manager (optional dependency)
try:
    from webdriver_manager.chrome import ChromeDriverManager
    from webdriver_manager.microsoft import EdgeChromiumDriverManager
    WEBDRIVER_MANAGER_AVAILABLE = True
except ImportError:
    WEBDRIVER_MANAGER_AVAILABLE = False


class BonolotoScraper:
    """Scrapes Bonoloto lottery results from the official website."""
    
    URL = "https://www.loteriasyapuestas.es/es/resultados/bonoloto"
    
    def __init__(self, headless: bool = True):
        """
        Initialize the scraper.
        
        Args:
            headless: Run browser in headless mode (no visible window)
        """
        self.headless = headless
        self.driver = None
        
    def _setup_driver(self):
        """Set up WebDriver - tries Edge first (Windows default), then Chrome as fallback."""
        # Try Edge first (typically pre-installed on Windows)
        try:
            self.driver = self._setup_edge()
            cf.printInfo("Using Edge browser", colorama.Fore.GREEN)
            return
        except Exception as e:
            cf.printInfo(f"Edge not available: {e}. Trying Chrome...", colorama.Fore.YELLOW)
        
        # Fallback to Chrome
        try:
            self.driver = self._setup_chrome()
            cf.printInfo("Using Chrome browser", colorama.Fore.GREEN)
            return
        except Exception as e:
            cf.printInfo(f"Chrome not available: {e}", colorama.Fore.RED)
            raise WebDriverException("No compatible browser found. Please install Chrome or Edge.")
    
    def _setup_chrome(self):
        """Set up Chrome WebDriver."""
        options = ChromeOptions()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Try direct Chrome first, fall back to webdriver-manager
        try:
            driver = webdriver.Chrome(options=options)
        except Exception:
            if WEBDRIVER_MANAGER_AVAILABLE:
                service = ChromeService(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=options)
            else:
                raise
        
        driver.implicitly_wait(10)
        return driver
    
    def _setup_edge(self):
        """Set up Edge WebDriver."""
        options = EdgeOptions()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        # Try direct Edge first, fall back to webdriver-manager
        try:
            driver = webdriver.Edge(options=options)
        except Exception:
            if WEBDRIVER_MANAGER_AVAILABLE:
                service = EdgeService(EdgeChromiumDriverManager().install())
                driver = webdriver.Edge(service=service, options=options)
            else:
                raise
        
        driver.implicitly_wait(10)
        return driver
        
    def _close_driver(self):
        """Close the WebDriver."""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def _accept_cookies(self):
        """Accept cookies if the banner appears."""
        try:
            cookie_btn = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
            )
            cookie_btn.click()
            cf.printInfo("Cookies accepted", colorama.Fore.GREEN)
        except TimeoutException:
            # Cookie banner not present or already accepted
            pass
    
    def _parse_date(self, date_str: str) -> str:
        """
        Parse date string from various formats to YYYY-MM-DD.
        
        Args:
            date_str: Date string like "04/12/2025" or "Viernes, 04 de diciembre de 2025"
            
        Returns:
            Date in YYYY-MM-DD format
        """
        # Try DD/MM/YYYY format first
        match = re.search(r'(\d{2})/(\d{2})/(\d{4})', date_str)
        if match:
            day, month, year = match.groups()
            return f"{year}-{month}-{day}"
        
        # Try to parse from long format like "Viernes, 04 de diciembre de 2025"
        months = {
            'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04',
            'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
            'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12'
        }
        
        match = re.search(r'(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})', date_str.lower())
        if match:
            day, month_name, year = match.groups()
            month = months.get(month_name, '01')
            return f"{year}-{month}-{day.zfill(2)}"
        
        return None
    
    def _extract_numbers(self, container) -> dict:
        """
        Extract numbers from a result container.
        
        Args:
            container: Selenium WebElement containing the result
            
        Returns:
            Dictionary with date and numbers, or None if parsing fails
        """
        try:
            # Find date - it can be in different elements
            date_str = None
            
            # Try finding date in various locations
            date_selectors = [
                ".cuerpoResultado .fecha",
                ".fechaSorteo",
                ".fecha-sorteo",
                "h3",
                ".tituloResultado"
            ]
            
            for selector in date_selectors:
                try:
                    date_elem = container.find_element(By.CSS_SELECTOR, selector)
                    date_str = self._parse_date(date_elem.text)
                    if date_str:
                        break
                except NoSuchElementException:
                    continue
            
            if not date_str:
                # Try to find any element with a date pattern
                all_text = container.text
                date_str = self._parse_date(all_text)
            
            if not date_str:
                return None
            
            # Find main numbers (6 numbers)
            numbers = []
            
            # Try different selectors for main numbers
            number_selectors = [
                ".combinacionPrincipal .numero",
                ".numeros-combinacion .numero",
                ".bola-numero",
                ".numero"
            ]
            
            for selector in number_selectors:
                try:
                    num_elements = container.find_elements(By.CSS_SELECTOR, selector)
                    if len(num_elements) >= 6:
                        numbers = [int(el.text.strip()) for el in num_elements[:6] if el.text.strip().isdigit()]
                        if len(numbers) == 6:
                            break
                except (NoSuchElementException, ValueError):
                    continue
            
            # If still no numbers, try regex on container text
            if len(numbers) != 6:
                all_text = container.text
                # Look for 6 consecutive 2-digit numbers
                number_matches = re.findall(r'\b(\d{1,2})\b', all_text)
                if len(number_matches) >= 6:
                    numbers = [int(n) for n in number_matches[:6] if 1 <= int(n) <= 49]
            
            if len(numbers) != 6:
                return None
            
            # Find complementario
            complementario = None
            comp_selectors = [
                ".complementario .numero",
                ".numero-complementario",
            ]
            
            for selector in comp_selectors:
                try:
                    comp_elem = container.find_element(By.CSS_SELECTOR, selector)
                    comp_text = comp_elem.text.strip()
                    if comp_text.isdigit():
                        complementario = int(comp_text)
                        break
                except (NoSuchElementException, ValueError):
                    continue
            
            # Find reintegro
            reintegro = None
            reint_selectors = [
                ".reintegro .numero",
                ".numero-reintegro",
            ]
            
            for selector in reint_selectors:
                try:
                    reint_elem = container.find_element(By.CSS_SELECTOR, selector)
                    reint_text = reint_elem.text.strip()
                    if reint_text.isdigit():
                        reintegro = int(reint_text)
                        break
                except (NoSuchElementException, ValueError):
                    continue
            
            # If comp/reintegro not found via selectors, try parsing structure
            if complementario is None or reintegro is None:
                all_text = container.text
                # Pattern: look for "C" followed by number for complementario
                comp_match = re.search(r'[Cc](?:omplementario)?[\s:]*(\d{1,2})', all_text)
                if comp_match:
                    complementario = int(comp_match.group(1))
                
                # Pattern: look for "R" or "Reintegro" followed by number
                reint_match = re.search(r'[Rr](?:eintegro)?[\s:]*(\d)', all_text)
                if reint_match:
                    reintegro = int(reint_match.group(1))
            
            return {
                "date": date_str,
                "numbers": numbers,
                "complementario": complementario,
                "reintegro": reintegro
            }
            
        except Exception as e:
            cf.printInfo(f"Error extracting numbers: {e}", colorama.Fore.YELLOW)
            return None
    
    def scrape_results(self, max_results: int = 10) -> list:
        """
        Scrape Bonoloto results from the official website.
        
        Args:
            max_results: Maximum number of results to fetch
            
        Returns:
            List of dictionaries with date and numbers
        """
        results = []
        
        try:
            cf.printInfo(f"Scraping Bonoloto results from: {self.URL}", colorama.Fore.CYAN)
            
            self._setup_driver()
            self.driver.get(self.URL)
            
            # Accept cookies
            self._accept_cookies()
            
            # Wait for results to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".cuerpoRegionResultado, .resultado-sorteo, .resultados"))
            )
            
            # Scroll to load more results
            for _ in range(3):
                self.driver.execute_script("window.scrollBy(0, 500)")
                import time
                time.sleep(0.5)
            
            # Find all result containers
            result_selectors = [
                ".cuerpoRegionResultado",
                ".resultado-sorteo",
                ".bloque-resultado"
            ]
            
            containers = []
            for selector in result_selectors:
                try:
                    containers = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if containers:
                        break
                except NoSuchElementException:
                    continue
            
            cf.printInfo(f"Found {len(containers)} result containers", colorama.Fore.GREEN)
            
            # Extract data from each container
            for container in containers[:max_results]:
                result = self._extract_numbers(container)
                if result and result["date"]:
                    results.append(result)
                    cf.printInfo(f"Extracted: {result['date']} - {result['numbers']} C:{result['complementario']} R:{result['reintegro']}", colorama.Fore.GREEN)
            
        except Exception as e:
            cf.printInfo(f"Error during scraping: {e}", colorama.Fore.RED)
            
        finally:
            self._close_driver()
        
        return results


def scrape_bonoloto(headless: bool = True, max_results: int = 10) -> list:
    """
    Convenience function to scrape Bonoloto results.
    
    Args:
        headless: Run browser without visible window
        max_results: Maximum number of results to fetch
        
    Returns:
        List of result dictionaries
    """
    scraper = BonolotoScraper(headless=headless)
    return scraper.scrape_results(max_results=max_results)


if __name__ == "__main__":
    # Test the scraper
    colorama.init()
    results = scrape_bonoloto(headless=False, max_results=5)
    print(f"\nScraped {len(results)} results:")
    for r in results:
        print(f"  {r['date']}: {r['numbers']} C:{r['complementario']} R:{r['reintegro']}")
