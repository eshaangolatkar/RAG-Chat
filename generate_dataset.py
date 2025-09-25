# generate_dataset.py - Improved version with domain-specific templates
import uuid
import random
import pandas as pd
from faker import Faker
import os

F = Faker()
os.makedirs("data", exist_ok=True)

ROLES = ["Founder", "Co-founder", "Engineer", "PM", "Investor", "Other"]
STAGES = ["none", "pre-seed", "seed", "series A", "growth"]
KEYWORDS_POOL = [
    "healthtech","AI","marketplace","fintech","edtech","analytics","saas",
    "hardware","biotech","cleantech","blockchain","consumer","mobility","iot","robotics"
]

# Indian cities list - covers major startup hubs and other cities
INDIAN_CITIES = [
    # Major startup hubs
    "Bangalore, India", "Bengaluru, India", "Mumbai, India", "Delhi, India", 
    "New Delhi, India", "Hyderabad, India", "Pune, India", "Chennai, India",
    "Gurgaon, India", "Gurugram, India", "Noida, India",
    
    # Other major cities
    "Kolkata, India", "Ahmedabad, India", "Jaipur, India", "Surat, India",
    "Lucknow, India", "Kanpur, India", "Nagpur, India", "Indore, India",
    "Thane, India", "Bhopal, India", "Visakhapatnam, India", "Pimpri-Chinchwad, India",
    "Patna, India", "Vadodara, India", "Ludhiana, India", "Agra, India",
    "Nashik, India", "Faridabad, India", "Meerut, India", "Rajkot, India",
    "Kalyan-Dombivli, India", "Vasai-Virar, India", "Varanasi, India", "Srinagar, India",
    "Aurangabad, India", "Dhanbad, India", "Amritsar, India", "Navi Mumbai, India",
    "Allahabad, India", "Ranchi, India", "Howrah, India", "Coimbatore, India",
    "Jabalpur, India", "Gwalior, India", "Vijayawada, India", "Jodhpur, India",
    "Madurai, India", "Raipur, India", "Kota, India", "Chandigarh, India",
    "Guwahati, India", "Solapur, India", "Hubli-Dharwad, India", "Tiruchirappalli, India",
    "Bareilly, India", "Mysore, India", "Tiruppur, India", "Ghaziabad, India",
    "Jalandhar, India", "Bhubaneswar, India", "Salem, India", "Warangal, India",
    
    # Tech/startup cities with variations
    "Kochi, India", "Cochin, India", "Thiruvananthapuram, India", "Trivandrum, India",
    "Mangalore, India", "Mysuru, India", "Coimbatore, India", "Vellore, India",
    "Puducherry, India", "Pondicherry, India", "Shimla, India", "Dehradun, India",
]

# Domain-specific idea templates
IDEA_TEMPLATES = {
    "AI": [
        "developing AI-powered solutions for automated business process optimization",
        "building machine learning models to predict customer behavior patterns",
        "creating intelligent chatbots for customer service automation",
        "developing computer vision systems for quality control in manufacturing",
        "building NLP tools for automated document analysis and extraction",
        "creating AI-driven recommendation engines for e-commerce platforms",
        "developing predictive analytics tools for supply chain management",
        "building deep learning models for fraud detection in financial transactions"
    ],
    "healthtech": [
        "developing digital health platforms for remote patient monitoring",
        "building telemedicine solutions for rural healthcare access",
        "creating AI-driven diagnostic tools for early disease detection",
        "developing wearable health devices for chronic condition management",
        "building electronic health record systems with smart analytics",
        "creating mental health apps with personalized therapy recommendations",
        "developing medication adherence tracking solutions for elderly patients",
        "building health data integration platforms for hospitals"
    ],
    "fintech": [
        "building digital payment solutions for small businesses and merchants",
        "developing blockchain-based lending platforms for underbanked populations",
        "creating automated investment tools for retail investors",
        "building expense management software for SMEs",
        "developing cryptocurrency exchange platforms with enhanced security",
        "creating peer-to-peer lending solutions for personal loans",
        "building robo-advisory services for wealth management",
        "developing digital banking solutions for rural communities"
    ],
    "marketplace": [
        "building B2B marketplaces connecting manufacturers with retailers",
        "developing peer-to-peer rental platforms for equipment sharing",
        "creating online marketplaces for local service providers",
        "building agricultural marketplaces connecting farmers to buyers",
        "developing freelance platforms for specialized professional services",
        "creating secondhand goods marketplaces with quality assurance",
        "building hyperlocal delivery marketplaces for groceries and essentials",
        "developing skill-based learning marketplaces for upskilling"
    ],
    "edtech": [
        "developing online learning platforms for professional skill development",
        "building AI-powered tutoring systems for K-12 education",
        "creating virtual reality training solutions for technical skills",
        "developing language learning apps with speech recognition technology",
        "building assessment and testing platforms for educational institutions",
        "creating coding bootcamp platforms with hands-on project learning",
        "developing corporate training solutions with gamification elements",
        "building platforms connecting students with industry mentors"
    ],
    "saas": [
        "building cloud-based CRM solutions for small and medium businesses",
        "developing project management tools with advanced collaboration features",
        "creating HR management platforms with automated recruitment processes",
        "building inventory management systems for retail and e-commerce",
        "developing customer support software with AI-powered ticket routing",
        "creating accounting and invoicing solutions for freelancers and agencies",
        "building marketing automation platforms for email and social media",
        "developing business intelligence dashboards with real-time analytics"
    ],
    "hardware": [
        "developing IoT sensors for smart home automation systems",
        "building wearable devices for fitness and health tracking",
        "creating drone technology for agricultural monitoring and spraying",
        "developing smart manufacturing equipment with predictive maintenance",
        "building electric vehicle charging infrastructure and management systems",
        "creating robotic process automation hardware for warehouses",
        "developing environmental monitoring devices for air and water quality",
        "building smart city infrastructure solutions for traffic management"
    ],
    "biotech": [
        "developing personalized medicine solutions based on genetic profiling",
        "building lab-on-chip devices for rapid diagnostic testing",
        "creating bioinformatics platforms for drug discovery research",
        "developing gene therapy delivery systems for rare diseases",
        "building synthetic biology platforms for sustainable chemical production",
        "creating tissue engineering solutions for regenerative medicine",
        "developing microbiome analysis tools for personalized nutrition",
        "building protein engineering platforms for therapeutic development"
    ],
    "cleantech": [
        "developing solar energy solutions for residential and commercial use",
        "building waste management systems with automated sorting technology",
        "creating carbon capture and storage solutions for industrial applications",
        "developing water purification systems for rural and urban communities",
        "building energy storage solutions using advanced battery technology",
        "creating smart grid systems for efficient electricity distribution",
        "developing biofuel production processes from agricultural waste",
        "building air pollution monitoring and mitigation systems"
    ],
    "blockchain": [
        "developing decentralized finance protocols for lending and borrowing",
        "building supply chain transparency platforms using blockchain technology",
        "creating NFT marketplaces for digital art and collectibles",
        "developing smart contract platforms for automated business processes",
        "building cryptocurrency wallets with enhanced security features",
        "creating blockchain-based identity verification systems",
        "developing decentralized storage solutions for enterprise data",
        "building tokenization platforms for real estate investments"
    ],
    "consumer": [
        "building direct-to-consumer brands for sustainable lifestyle products",
        "developing subscription box services for niche hobby communities",
        "creating personalized nutrition and meal planning applications",
        "building social commerce platforms for influencer-driven sales",
        "developing smart home appliances with voice control integration",
        "creating fashion recommendation apps using AI and computer vision",
        "building community-driven platforms for local event discovery",
        "developing wellness and mindfulness apps for stress management"
    ],
    "mobility": [
        "developing ride-sharing platforms for intercity transportation",
        "building electric scooter sharing systems for urban mobility",
        "creating route optimization software for logistics and delivery",
        "developing autonomous vehicle testing and simulation platforms",
        "building bike-sharing systems with IoT-enabled smart locks",
        "creating parking management solutions for smart cities",
        "developing fleet management software for commercial vehicles",
        "building public transit optimization systems using real-time data"
    ],
    "iot": [
        "developing smart agriculture monitoring systems for crop optimization",
        "building industrial IoT platforms for manufacturing efficiency",
        "creating connected healthcare devices for remote patient monitoring",
        "developing smart city infrastructure with sensor networks",
        "building home automation systems with energy management features",
        "creating asset tracking solutions for supply chain visibility",
        "developing environmental monitoring networks for pollution control",
        "building predictive maintenance systems for industrial equipment"
    ],
    "robotics": [
        "developing autonomous robots for warehouse automation and fulfillment",
        "building surgical robotics systems for minimally invasive procedures",
        "creating educational robotics platforms for STEM learning",
        "developing service robots for elderly care and assistance",
        "building agricultural robots for planting, harvesting, and monitoring",
        "creating inspection robots for hazardous environment monitoring",
        "developing rehabilitation robotics for physical therapy applications",
        "building cleaning and maintenance robots for commercial facilities"
    ],
    "analytics": [
        "building business intelligence platforms for data-driven decision making",
        "developing customer analytics tools for e-commerce optimization",
        "creating predictive maintenance analytics for industrial equipment",
        "building social media analytics platforms for brand monitoring",
        "developing financial analytics tools for investment portfolio management",
        "creating sports analytics platforms for team performance optimization",
        "building healthcare analytics systems for population health management",
        "developing marketing attribution analytics for multi-channel campaigns"
    ]
}

# Domain-specific background templates
BACKGROUND_TEMPLATES = {
    "AI": [
        "computer science with specialization in machine learning and neural networks",
        "data science with extensive experience in deep learning frameworks",
        "software engineering with focus on AI/ML system architecture",
        "research in artificial intelligence and natural language processing"
    ],
    "healthtech": [
        "biomedical engineering with experience in medical device development",
        "healthcare administration with digital health transformation experience",
        "medicine with focus on healthcare technology and informatics",
        "biotechnology with experience in clinical research and development"
    ],
    "fintech": [
        "finance with expertise in digital banking and payment systems",
        "economics with specialization in blockchain and cryptocurrency",
        "computer science with experience in financial software development",
        "business with focus on financial services and regulatory compliance"
    ],
    "marketplace": [
        "business development with experience in platform economics",
        "e-commerce with expertise in marketplace operations and growth",
        "product management with focus on two-sided market dynamics",
        "operations with experience in supply chain and logistics"
    ]
}

# Company name prefixes that match the domain
COMPANY_PREFIXES = {
    "AI": ["Neural", "Cognitive", "Smart", "Intel", "Auto", "Predict", "Learn", "Brain"],
    "healthtech": ["Care", "Health", "Med", "Bio", "Vital", "Cure", "Heal", "Life"],
    "fintech": ["Pay", "Fin", "Money", "Cash", "Credit", "Invest", "Wealth", "Bank"],
    "marketplace": ["Connect", "Market", "Trade", "Exchange", "Link", "Hub", "Bridge", "Network"],
    "edtech": ["Learn", "Edu", "Skill", "Study", "Teach", "Know", "Train", "Academy"],
    "saas": ["Cloud", "Soft", "Tech", "Pro", "Sys", "Work", "Flow", "Manage"],
    "hardware": ["Tech", "Device", "Smart", "IoT", "Sensor", "Circuit", "Build", "Maker"],
    "biotech": ["Bio", "Gene", "Life", "Cell", "Lab", "Pharma", "Protein", "DNA"],
    "cleantech": ["Green", "Clean", "Eco", "Solar", "Energy", "Sustain", "Carbon", "Environment"],
    "blockchain": ["Block", "Chain", "Crypto", "Defi", "Web3", "Token", "Decentralized", "Ledger"],
    "consumer": ["Brand", "Direct", "Consumer", "Lifestyle", "Personal", "Custom", "Social", "Trend"],
    "mobility": ["Move", "Transit", "Ride", "Transport", "Mobile", "Journey", "Route", "Go"],
    "iot": ["Connect", "Smart", "IoT", "Sensor", "Device", "Network", "Monitor", "Remote"],
    "robotics": ["Robo", "Auto", "Machine", "Bot", "Mech", "Autonomous", "AI", "Intelligent"],
    "analytics": ["Data", "Analytics", "Insights", "Intelligence", "Metrics", "Track", "Measure", "Report"]
}

COMPANY_SUFFIXES = ["Labs", "Tech", "Solutions", "Systems", "Platform", "AI", "Pro", "Works", "Hub", "Studio"]

def random_keywords():
    """Generate 1-3 related keywords"""
    k = random.sample(KEYWORDS_POOL, k=random.choice([1,2,3]))
    return ", ".join(k)

def create_realistic_idea(keywords):
    """Create a realistic startup idea based on keywords"""
    primary_keyword = keywords.split(',')[0].strip()
    
    if primary_keyword in IDEA_TEMPLATES:
        return random.choice(IDEA_TEMPLATES[primary_keyword])
    else:
        # Fallback for keywords not in templates
        return f"developing innovative {primary_keyword} solutions for enterprise clients"

def create_realistic_about(name, keywords, role):
    """Create a realistic founder background"""
    primary_keyword = keywords.split(',')[0].strip()
    
    # Get background template
    if primary_keyword in BACKGROUND_TEMPLATES:
        background = random.choice(BACKGROUND_TEMPLATES[primary_keyword])
    else:
        background = random.choice(["computer science", "business", "engineering", "product management"])
    
    # Create experience description
    experiences = [
        f"previously led product development at {F.company()}",
        f"worked as senior engineer at {F.company()}",
        f"was VP of Engineering at {F.company()}",
        f"founded and sold a {primary_keyword} startup",
        f"worked in consulting focusing on {primary_keyword} implementations",
        f"led growth initiatives at a Series B {primary_keyword} company"
    ]
    
    experience = random.choice(experiences)
    
    # Create current focus
    focuses = [
        f"passionate about solving real-world problems through {primary_keyword}",
        f"focused on building scalable {primary_keyword} solutions",
        f"dedicated to democratizing access to {primary_keyword} technology",
        f"committed to driving innovation in the {primary_keyword} space"
    ]
    
    focus = random.choice(focuses)
    
    return f"{name} has a background in {background}, {experience}, and is {focus}."

def create_realistic_company_name(keywords):
    """Create a company name that matches the domain"""
    primary_keyword = keywords.split(',')[0].strip()
    
    if primary_keyword in COMPANY_PREFIXES:
        prefix = random.choice(COMPANY_PREFIXES[primary_keyword])
    else:
        prefix = random.choice(["Tech", "Smart", "Pro", "Digital"])
    
    suffix = random.choice(COMPANY_SUFFIXES)
    
    return f"{prefix}{suffix}"

def make_row():
    """Generate a single realistic founder profile"""
    name = F.name()
    keywords = random_keywords()
    role = random.choice(ROLES)
    company = create_realistic_company_name(keywords)
    
    return {
        "id": str(uuid.uuid4()),
        "founder_name": name,
        "email": f"{name.lower().replace(' ', '_')}@example.com",
        "role": role,
        "company": company,
        "location": random.choice(INDIAN_CITIES),
        "idea": create_realistic_idea(keywords),
        "about": create_realistic_about(name, keywords, role),
        "keywords": keywords,
        "stage": random.choice(STAGES),
        "linked_in": f"https://www.linkedin.com/in/",
    }

def generate_csv(out_path="data/founders.csv", n=700):
    print("Starting improved dataset generation with domain-specific templates...")
    rows = [make_row() for _ in range(n)]
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(df)} rows")
    
    # Print location distribution
    location_counts = df['location'].value_counts()
    print(f"\nLocation distribution (top 10):")
    print(location_counts.head(10))
    print(f"\nTotal unique locations: {len(location_counts)}")
    
    # Print keyword distribution
    all_keywords = []
    for keywords in df['keywords']:
        all_keywords.extend([k.strip() for k in keywords.split(',')])
    
    from collections import Counter
    keyword_counts = Counter(all_keywords)
    print(f"\nKeyword distribution (top 10):")
    for keyword, count in keyword_counts.most_common(10):
        print(f"  {keyword}: {count}")
    
    # Print sample profiles to inspect quality
    print("\nSample realistic profiles:")
    print("="*80)
    for i in range(3):
        row = df.iloc[i]
        print(f"Founder: {row['founder_name']} ({row['role']})")
        print(f"Company: {row['company']}")
        print(f"Location: {row['location']}")
        print(f"Keywords: {row['keywords']}")
        print(f"Idea: {row['idea']}")
        print(f"About: {row['about']}")
        print("-" * 80)
    
    print(f"\nImproved dataset generation complete! File saved to: {out_path}")

if __name__ == "__main__":
    generate_csv()
