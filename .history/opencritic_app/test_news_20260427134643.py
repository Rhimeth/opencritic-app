from news import GamingNewsService

service = GamingNewsService(enable_deduplication=True)
news = service.fetch(per_source=3, deduplicate=True)

print(f"Fetched {len(news)} unique news items\n")
for item in news[:5]:
    print(f"📰 {item.source}: {item.title}")
    print(f"   Summary: {item.summary[:120]}...")
    print(f"   Link: {item.link}\n")