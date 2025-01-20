from playwright.sync_api import sync_playwright

def crawl_musinsa():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        url = "https://www.musinsa.com/snap/main/recommend"
        page.goto(url)

        page.screenshot(path="screenshot.png")
        browser.close()