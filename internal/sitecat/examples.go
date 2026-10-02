package sitecat

// Examples are typical sites of each category, written the way the
// embedding classifier sees a site ("host title"). Together with each
// category's label and description they are the prototypes a site is
// compared with. Keep them distinct from the evaluation set in
// internal/ml/classify/testdata/sites.tsv, or the reported accuracy means
// nothing.
var Examples = map[string][]string{
	"adult": {
		"pornhub.com Free Porn Videos & Sex Movies",
		"xvideos.com Free Porn Videos - XVIDEOS",
		"onlyfans.com OnlyFans creators adult content subscriptions",
		"escort-directory.example Escorts and adult services near you",
		"hentai-haven.example Uncensored hentai anime",
	},
	"dating": {
		"tinder.com Tinder | Dating, Make Friends & Meet New People",
		"match.com Match.com: online dating site for singles",
		"okcupid.com OkCupid | Online Dating App for Great Dates",
		"bumble.com Bumble - Date, Meet Friends, Network",
	},
	"gambling": {
		"bet365.com bet365 - Online Sports Betting",
		"pokerstars.com PokerStars online poker tournaments",
		"draftkings.com DraftKings sportsbook and casino",
		"casino-online.example Play slots, roulette and blackjack for real money",
		"lottery.example National Lottery results and tickets",
	},
	"social_media": {
		"facebook.com Facebook - log in or sign up",
		"instagram.com Instagram photos and reels",
		"tiktok.com TikTok - Make Your Day",
		"x.com X. It's what's happening",
		"reddit.com Reddit - The heart of the internet",
		"pinterest.com Pinterest ideas and pins",
	},
	"chat_messaging": {
		"discord.com Discord - group chat that's all fun & games",
		"web.whatsapp.com WhatsApp Web",
		"web.telegram.org Telegram Web messenger",
		"slack.com Slack team chat and messaging",
	},
	"email": {
		"mail.google.com Gmail inbox",
		"outlook.live.com Outlook mail - Sign in",
		"mail.yahoo.com Yahoo Mail",
		"proton.me Proton Mail: secure encrypted email",
	},
	"news": {
		"nytimes.com The New York Times - Breaking News, US News, World News",
		"theguardian.com News, sport and opinion from the Guardian",
		"cnn.com CNN: Breaking News, Latest News and Videos",
		"reuters.com Reuters | Breaking International News & Views",
		"lemonde.fr Le Monde.fr - Actualités et Infos",
	},
	"shopping": {
		"amazon.com Amazon.com. Spend less. Smile more.",
		"ebay.com Electronics, Cars, Fashion, Collectibles & More | eBay",
		"etsy.com Etsy - Shop for handmade, vintage, custom, and unique gifts",
		"walmart.com Walmart | Save Money. Live better.",
		"ikea.com IKEA furniture and home accessories",
	},
	"banking_finance": {
		"bankofamerica.com Bank of America - Banking, Credit Cards, Loans",
		"paypal.com PayPal: pay, send money and accept payments",
		"coinbase.com Coinbase - Buy and sell Bitcoin, Ethereum",
		"fidelity.com Fidelity Investments - retirement planning, brokerage",
		"geico.com GEICO car insurance quotes",
	},
	"streaming_video": {
		"youtube.com YouTube",
		"netflix.com Netflix - Watch TV Shows Online, Watch Movies Online",
		"twitch.tv Twitch live streaming",
		"disneyplus.com Disney+ streaming movies and series",
		"vimeo.com Vimeo video hosting",
	},
	"music_audio": {
		"open.spotify.com Spotify - Web Player: Music for everyone",
		"soundcloud.com SoundCloud - Hear the world's sounds",
		"music.apple.com Apple Music",
		"tunein.com TuneIn radio stations and podcasts",
	},
	"gaming": {
		"roblox.com Roblox games",
		"epicgames.com Epic Games Store - Download & Play PC Games",
		"minecraft.net Minecraft official site",
		"ign.com IGN video game news and reviews",
		"chess.com Chess.com - Play Chess Online",
	},
	"entertainment": {
		"imdb.com IMDb: Ratings, Reviews, and Where to Watch the Best Movies & TV Shows",
		"tmz.com TMZ celebrity gossip and entertainment news",
		"9gag.com 9GAG - Best Funny Memes",
		"rottentomatoes.com Rotten Tomatoes: movies, TV shows, reviews",
	},
	"sports": {
		"espn.com ESPN - Serving Sports Fans",
		"nba.com NBA official site: scores, schedule, stats",
		"skysports.com Sky Sports football, cricket, F1",
		"uefa.com UEFA Champions League",
	},
	"search": {
		"google.com Google search",
		"bing.com Bing",
		"duckduckgo.com DuckDuckGo - Protection. Privacy. Peace of mind.",
		"yahoo.com Yahoo | Mail, Weather, Search, Politics, News",
	},
	"education": {
		"wikipedia.org Wikipedia, the free encyclopedia",
		"khanacademy.org Khan Academy free online courses and lessons",
		"coursera.org Coursera online courses from top universities",
		"mit.edu Massachusetts Institute of Technology",
		"duolingo.com Duolingo - learn a language for free",
	},
	"kids": {
		"pbskids.org PBS KIDS games and videos for children",
		"nickjr.com Nick Jr. games for preschoolers",
		"coolmathgames.com Cool Math Games for kids",
		"abcya.com ABCya educational games for kids",
	},
	"government": {
		"irs.gov Internal Revenue Service",
		"gov.uk Welcome to GOV.UK",
		"usa.gov Official guide to government information and services",
		"europa.eu European Union official website",
	},
	"health": {
		"webmd.com WebMD - Better information. Better health.",
		"mayoclinic.org Mayo Clinic: diseases, conditions and treatment",
		"cvs.com CVS pharmacy prescriptions",
		"myfitnesspal.com MyFitnessPal calorie counter and fitness",
	},
	"travel": {
		"booking.com Booking.com: hotels, apartments and flights",
		"expedia.com Expedia travel: vacation homes, hotels, car rentals, flights",
		"airbnb.com Airbnb vacation rentals",
		"delta.com Delta Air Lines flights and tickets",
		"maps.google.com Google Maps directions",
	},
	"jobs": {
		"indeed.com Indeed job search",
		"linkedin.com/jobs LinkedIn jobs and careers",
		"glassdoor.com Glassdoor company reviews and salaries",
		"monster.com Monster job search and recruiting",
	},
	"religion": {
		"vatican.va The Holy See",
		"biblegateway.com BibleGateway - search the Bible online",
		"quran.com The Noble Quran",
		"chabad.org Judaism, Torah and Jewish info",
	},
	"technology": {
		"github.com GitHub: where the world builds software",
		"stackoverflow.com Stack Overflow developer questions and answers",
		"theverge.com The Verge technology news",
		"microsoft.com Microsoft - cloud, computers, apps and gaming",
		"python.org Welcome to Python.org",
	},
	"business": {
		"salesforce.com Salesforce CRM software",
		"zoom.us Zoom video meetings",
		"office.com Microsoft 365 Office apps",
		"notion.so Notion workspace for notes and docs",
		"acme-industrial.example Acme Industrial Supplies - company profile and contact",
	},
	"ads_tracking": {
		"doubleclick.net DoubleClick ad serving",
		"googlesyndication.com Google AdSense ad syndication",
		"google-analytics.com Google Analytics tracking",
		"adnxs.com AppNexus ad exchange",
		"scorecardresearch.com Scorecard Research audience measurement",
	},
	"malware_phishing": {
		"paypal-account-verify.example Verify your PayPal account now - suspended",
		"secure-login-appleid.example Apple ID locked - confirm your identity",
		"free-iphone-winner.example Congratulations you won a free iPhone, claim now",
		"crack-download.example Download keygen crack exe",
	},
	"piracy": {
		"thepiratebay.org The Pirate Bay - torrents",
		"1337x.to 1337x free movie and game torrents",
		"fmovies.example Watch free movies online in HD, no sign up",
		"rarbg.example RARBG torrents for movies and TV",
	},
	"drugs_alcohol": {
		"leafly.com Leafly cannabis strains and dispensaries",
		"drizly.com Drizly alcohol delivery: beer, wine and liquor",
		"vape-shop.example Vape shop e-liquids and vaporizers",
		"erowid.org Erowid psychoactive drug information",
	},
	"weapons": {
		"gunbroker.com GunBroker guns for sale",
		"budsgunshop.com Buds Gun Shop firearms and ammunition",
		"knifecenter.com KnifeCenter knives and swords",
		"ammo-depot.example Bulk ammunition for sale",
	},
	"violence_hate": {
		"bestgore.example Real gore videos and death footage",
		"extremist-forum.example White power forum",
		"hate-speech.example Racist propaganda against immigrants",
	},
	Infrastructure: {
		"cdn.jsdelivr.net jsDelivr CDN",
		"fonts.gstatic.com",
		"api.stripe.com",
		"login.microsoftonline.com Sign in to your account",
		"s3.amazonaws.com",
		"cloudfront.net",
		"update.googleapis.com",
	},
	Other: {
		"example.com Example Domain",
		"johnsmith.example John Smith personal homepage",
		"parked-domain.example This domain is for sale",
	},
}

// PrototypeTexts returns, per slug in display order, the texts a site is
// compared with: "Label: description" followed by the Examples.
func PrototypeTexts() []Prototype {
	out := make([]Prototype, 0, len(all))
	for _, c := range all {
		texts := append([]string{c.Label + ": " + c.Description}, Examples[c.Slug]...)
		out = append(out, Prototype{Slug: c.Slug, Texts: texts})
	}
	return out
}

// Prototype is one category's comparison texts.
type Prototype struct {
	Slug  string
	Texts []string
}
