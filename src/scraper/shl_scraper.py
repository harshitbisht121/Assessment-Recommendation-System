import pandas as pd
from playwright.sync_api import sync_playwright
import time

BASE = "https://www.shl.com/products/product-catalog/?start="
BASE_DOMAIN = "https://www.shl.com"

def scrape_catalog():
    records = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        page_size = 12
        
        # Scrape both product types
        for product_type in [1, 2]:  # 1 = Individual Test Solutions, 2 = Pre-packaged Job Solutions
            print(f"\n{'='*60}")
            print(f"SCRAPING TYPE {product_type}: {'Individual Test Solutions' if product_type == 1 else 'Pre-packaged Job Solutions'}")
            print(f"{'='*60}")
            
            empty_pages = 0
            page_no = 0
            
            while True:
                start = page_no * page_size
                url = f"{BASE}{start}&type={product_type}"
                
                print(f"\nLoading page {page_no+1}: {url}")
                
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=60000)
                    time.sleep(2)  # Wait for dynamic content
                    
                    # Find all product links in the table
                    # Look for links within table rows that go to product pages
                    product_links = page.query_selector_all("table tbody tr td a[href*='/products/product-catalog/view/']")
                    
                    if not product_links:
                        print("⚠ No product links found")
                        empty_pages += 1
                        if empty_pages >= 3:
                            print("No more pages for this type. Moving on.")
                            break
                        page_no += 1
                        continue
                    else:
                        empty_pages = 0
                    
                    # Extract unique product URLs
                    links = []
                    seen = set()
                    for link in product_links:
                        href = link.get_attribute("href")
                        if href and href not in seen:
                            if href.startswith("/"):
                                href = BASE_DOMAIN + href
                            links.append(href)
                            seen.add(href)
                    
                    print(f"Found {len(links)} unique product links")
                    
                    # Visit each product page
                    for idx, link in enumerate(links, 1):
                        try:
                            print(f"  [{idx}/{len(links)}] Visiting: {link}")
                            page.goto(link, wait_until="domcontentloaded", timeout=60000)
                            time.sleep(1)
                            
                            # Extract product name
                            try:
                                name = page.locator("h1").first.inner_text().strip()
                            except:
                                name = "N/A"
                            
                            # Extract description from meta tag
                            try:
                                desc = page.locator("meta[name='description']").get_attribute("content") or ""
                            except:
                                desc = ""
                            
                            # Get page content to determine test type
                            text = page.content().lower()
                            
                            # Determine test type based on content
                            test_types = []
                            if "personality" in text or "behavior" in text or "behaviour" in text:
                                test_types.append("P")
                            if "cognitive" in text or "ability" in text or "aptitude" in text:
                                test_types.append("A")
                            if "knowledge" in text or "skill" in text:
                                test_types.append("K")
                            if "simulation" in text:
                                test_types.append("S")
                            if "situational judgment" in text or "biodata" in text:
                                test_types.append("B")
                            
                            # If no type detected, default to K for Individual, P for Pre-packaged
                            if not test_types:
                                test_types = ["K"] if product_type == 1 else ["P"]
                            
                            records.append({
                                "name": name,
                                "url": link,
                                "description": desc,
                                "product_type": "Individual Test Solutions" if product_type == 1 else "Pre-packaged Job Solutions",
                                "duration": 40,
                                "remote_support": "Yes",
                                "adaptive_support": "Yes",
                                "test_type": test_types
                            })
                            
                            print(f"    ✔ {name} - Types: {', '.join(test_types)}")
                            
                        except Exception as e:
                            print(f"    ✘ Failed: {e}")
                    
                    page_no += 1
                    
                except Exception as e:
                    print(f"Error loading page: {e}")
                    empty_pages += 1
                    if empty_pages >= 3:
                        break
                    page_no += 1
        
        browser.close()
    
    # Create DataFrame and remove duplicates
    df = pd.DataFrame(records).drop_duplicates(subset=["url"])
    
    print("\n" + "="*60)
    print("FINAL COUNTS")
    print("="*60)
    print(f"Total scraped: {len(df)}")
    print(f"  - Individual Test Solutions: {len(df[df['product_type'] == 'Individual Test Solutions'])}")
    print(f"  - Pre-packaged Job Solutions: {len(df[df['product_type'] == 'Pre-packaged Job Solutions'])}")
    
    # Save to CSV
    df.to_csv("data/shl_catalog.csv", index=False)
    print("\n✓ Saved → data/shl_catalog.csv")
    
    return df


if __name__ == "__main__":
    scrape_catalog()