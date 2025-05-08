
from transformers import pipeline
import json
from tkinter import *
model_name = "deepset/bert-large-uncased-whole-word-masking-squad2"
context="""In the quiet town of Eldoria, nestled between rolling hills and a sparkling river, lived a young artist named Maya. Eldoria was known for its vibrant culture and picturesque landscapes, which inspired many artists like Maya. She spent her days painting the town's beautiful scenery, capturing the essence of its charm on her canvas.

One sunny afternoon, Maya decided to venture into the nearby forest, a place she had often admired from a distance but never explored. As she walked deeper into the woods, she came across a hidden grove where an ancient, majestic willow tree stood. Its long, sweeping branches created a natural canopy, and sunlight filtered through the leaves, casting a magical glow on the ground.

Maya felt a sudden surge of inspiration and set up her easel to paint the enchanting scene. Hours passed, and as dusk approached, she finished her masterpiece. Just as she was packing up, she noticed something glinting at the base of the willow tree. Curiosity piqued, she knelt down and found a small, intricately carved wooden box.

Carefully opening the box, Maya discovered a collection of old letters and a delicate locket. The letters, written in elegant handwriting, told the story of a couple who had once met under this very tree. Their love story was one of passion and adventure, but they had been separated by circumstances beyond their control. The locket contained a tiny portrait of the couple, forever young and in love.

Moved by the story and the profound sense of connection to the past, Maya decided to honor the couple by painting their story. She created a series of artworks depicting their journey, from their first meeting to their heartbreaking farewell. These paintings soon became the talk of Eldoria, touching the hearts of all who saw them.

Years later, Maya's paintings were displayed in a grand exhibition, and people from all over came to see the beautiful love story captured through her art. The ancient willow tree in the forest became a symbol of eternal love and inspiration, drawing visitors who wished to experience its magic for themselves.
"""
print(context)
#tkinter
root=Tk()
Label(root,text="Question answering chatbot").pack()
Label(root,text="Conext:"+context).pack()
Label(root,text="Enter your question:").pack()
question_=StringVar()
questionq=Entry(root,textvariable=question_).pack()
question=question_.get()

root.mainloop()
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = {
    'question': question,
    'context': context}
res = nlp(QA_input)
Label(root,text= res["answer"]).pack()
print("Answer:",res["answer"])
print("Starting Index:",res["start"])
print("Ending Index:",res["end"])
print("Accuracy:",res["score"])
output_data = {
    "context": context,
    "results": []
}
output_data["results"].append({
        "question": question,
        "answer": res["answer"],
        "start": res["start"],
        "end": res["end"],
        "score": res["score"]
    })
with open('output.json', 'w') as f:
    json.dump(output_data, f, indent=4)

print(res)

