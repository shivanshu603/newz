import asyncio
from news_monitor import NewsMonitor

async def main():
    monitor = NewsMonitor()
    await monitor.initialize()

if __name__ == "__main__":
    asyncio.run(main())
