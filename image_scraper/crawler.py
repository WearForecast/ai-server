import os
import aiohttp
import asyncio
from playwright.async_api import async_playwright

async def download_image(image_url, path):
    try: 
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                with open(path, "wb") as f:
                    f.write(await response.read())
    except Exception as e:
        print(f"Failed to download {image_url} - {e}")

async def crawl_musinsa_by_season(season_code):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto("https://www.musinsa.com/")

        base_url = "https://www.musinsa.com/snap/main/recommend"
        url = f"{base_url}?brands=&category=&genders=&height-range=&seasons={season_code}&styles=&tpos=&types=CODISHOP_SNAP&weight-range="
        
        try:
            await page.goto(url)
        except Exception as e:
            print(f"Failed to load page {url} - {e}")
            return
        
        season_dir = os.path.join("images", season_code)
        os.makedirs(season_dir, exist_ok=True)

        max_scroll_attempts = 3
        downloaded_images = 0

        for attempt in range(max_scroll_attempts):
            img_locator = page.locator("img[class='max-w-full w-full absolute m-auto inset-0 h-auto z-0 visible object-cover']")
            total_count = await img_locator.count()

            for i in range(total_count):
                src = await img_locator.nth(i).get_attribute("src")

                if src and src.startswith("http"):
                    filename = os.path.join(season_dir, f"{season_code}_{downloaded_images}.jpg")
                    src = src.replace("w=390", "w=1000")
                    print(f"Downloading image {downloaded_images + 1}: {src}")
                    await download_image(src, filename)
                    downloaded_images += 1
            
            print(f"Scrolling down... ({attempt + 1}/{max_scroll_attempts})")
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)

        await browser.close()

async def main():
    seasons = ["4", "3", "2", "1"]
    # 4: Spring, 3: Summer, 2: Fall, 1: Winter

    for season in seasons:
        await crawl_musinsa_by_season(season)

if __name__ == "__main__":
    asyncio.run(main())