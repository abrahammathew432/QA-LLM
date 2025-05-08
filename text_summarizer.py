from transformers import pipeline

# Load the summarization pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", framework="pt")

# prompt engineering
few_shot_prompts = [
    {
        "text": """Paragraph 1: Advancements in renewable energy technology have made it possible to harness energy from natural resources more efficiently than ever before. Solar panels have become more efficient and affordable, allowing homeowners to generate their own electricity and reduce reliance on fossil fuels.
Paragraph 2: Wind turbines are being installed in greater numbers, both onshore and offshore, contributing to the diversification of energy sources. Additionally, advancements in battery storage technology have addressed the intermittent nature of renewable energy sources, making it possible to store excess energy for use when the sun isn't shining or the wind isn't blowing.
Paragraph 3: These developments have not only helped reduce greenhouse gas emissions but have also created new jobs and stimulated economic growth in the renewable energy sector. As the world continues to grapple with the challenges of climate change, the importance of investing in and adopting renewable energy technologies cannot be overstated.""",
        "summary": "Renewable energy advancements have improved efficiency and affordability, leading to increased solar and wind energy use, better energy storage, and job creation."
    },
    {
        "text": """Paragraph 1: The development of the internet has transformed the way people communicate, access information, and conduct business. In the early days, the internet was primarily used for academic and research purposes, but it has since evolved into a global network that connects billions of people.
Paragraph 2: Social media platforms have revolutionized how we interact with one another, allowing for instant communication and the sharing of information on a scale never seen before. E-commerce has also flourished, with online shopping becoming a convenient alternative to traditional brick-and-mortar stores.
Paragraph 3: Furthermore, the internet has made it easier for people to access a wealth of information and educational resources, breaking down barriers to learning and fostering greater knowledge sharing. However, this rapid expansion of the internet has also brought challenges, such as issues related to privacy, security, and the spread of misinformation.""",
        "summary": "The internet has revolutionized communication, business, and access to information, despite challenges like privacy and misinformation."
    },
    {
        "text": """Paragraph 1: Artificial intelligence (AI) is rapidly advancing and finding applications across various industries. In healthcare, AI is being used to analyze medical data and assist in diagnosing diseases. AI algorithms can process vast amounts of data quickly and accurately, leading to more efficient and effective treatment plans.
Paragraph 2: In the automotive industry, AI is at the heart of self-driving car technology. Autonomous vehicles use AI to navigate roads, interpret traffic signals, and make real-time decisions to ensure passenger safety. This technology promises to reduce accidents and improve transportation efficiency.
Paragraph 3: The use of AI in finance has also grown, with algorithms being employed to detect fraudulent transactions, manage investment portfolios, and provide personalized financial advice. While AI offers numerous benefits, it also raises ethical concerns, such as job displacement and the need for transparent decision-making processes.""",
        "summary": "AI is advancing in healthcare, automotive, and finance industries, improving efficiency and decision-making but raising ethical concerns."
    }
]

# Text to summarize
text_to_summarize = """TIn the grand tapestry of existence, where galaxies spiral in cosmic ballets and stars twinkle like distant lanterns in the night sky, lies a universe of unfathomable wonders waiting to be explored. Across the vast expanse of space, celestial bodies of all shapes and sizes dance to the silent melody of gravity, weaving intricate patterns against the velvet backdrop of the cosmos. Nebulae, the cosmic nurseries where stars are born, glow with ethereal beauty, their swirling clouds of gas and dust giving birth to new suns and planets. Planetary systems, each a miniature solar system in its own right, orbit their parent stars in delicate celestial ballets, while moons, the silent sentinels of the night, cast their silvery glow upon alien landscapes below. And amidst it all, black holes lurk like silent behemoths, their gravitational pull distorting the very fabric of space and time. Supernovae, the cosmic explosions that mark the violent deaths of massive stars, illuminate the darkness with their fiery brilliance, scattering the building blocks of life across the cosmos. In this vast and wondrous universe, humanity stands as a curious observer, peering into the depths of space with wonder and awe, seeking to unlock the secrets of the cosmos and understand the mysteries of existence itself. From the smallest subatomic particles to the largest galaxies, the universe is a realm of infinite possibilities, where the laws of physics govern the motion of celestial bodies and the evolution of life. It is a place of beauty and chaos, where cosmic forces collide and stars are born in the fiery furnaces of stellar nurseries. It is a realm of time and space, where the past, present, and future coexist in a dance of eternal flux. And it is a domain of wonder and discovery, where the human spirit soars on wings of imagination, reaching out to touch the stars and explore the unknown. In this vast and wondrous universe, there are worlds beyond imagining, waiting to be explored and understood. And as we journey into the depths of space, we carry with us the dreams and aspirations of countless generations, united in our quest to unlock the secrets of the cosmos and chart a course for humanity among the stars.
"""# Generating summary
summary = summarizer(text_to_summarize, max_length=50, min_length=30, do_sample=False)[0]['summary_text']

print("Original Text:")
print(text_to_summarize)
print("\nSummary:")
print(summary)
