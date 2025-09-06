# GPT Implementation from Scratch 🚀  

A minimal yet educational implementation of **GPT (Generative Pretrained Transformer)** from scratch using **PyTorch**.  

This repo walks through how GPT works step by step — from **self-attention** to **multi-head attention**, **transformer blocks**, and finally, **text generation** at both **character-level** and **word-level**.  

---

## 📂 Project Structure  

```
GPT-implementation_from_scratch/
│── Explaining/               # Images & text explanations of GPT components
│── GPT/                      # Core GPT implementation
│   ├── single_head.py
│   ├── multi_head.py
│   ├── transformer_block.py
│   └── Gpt_class.py
│── Character_level/          # Character-level GPT (training attempted, no results saved)
│   ├── solution.py           # Generate function (character-level)
│   └── testing.py
│── Word_level/               # Word-level GPT (trained model + outputs)
│   ├── solution.py           # Generate function (word-level)
│   ├── testing.py            # Training + generation script
│   └── outputs/              # Screenshots of sample outputs
│── Doc_reader/               # Utilities to read input data
│   ├── doc_up.py             # Read from .docx
│   └── pdf_up.py             # Read from .pdf
│── Conversation.docx         # Training data (sample conversation text)
│── dcmachinebook.pdf         # Training data(DC Machines book, ~17k chars until ch1)
│── requirements.txt          # Dependencies of this project
│── wrd_trained_model.pkl     # Pretrained Model (for dat until ch1)
│── README.md                 # Project documentation
```

---

## ⚙️ Requirements  

Install dependencies:  

```bash
pip install torch torchtyping python-docx PyMuPDF
OR
pip install -r requirements.txt
```

📌 Notes:  
- `torch` → Core PyTorch library  
- `torchtyping` → For type annotations with tensors  
- `python-docx` → Read `.docx` files  
- `PyMuPDF` (`pip install PyMuPDF`) → Read `.pdf` files (via `import fitz`)  
- Standard libraries (`os`, etc.) are already included in Python.  


---

## 🏋️ Training  

To train the **word-level GPT** from scratch:  

```bash
python -m Word_level.testing
```

This will:  
1. Read training data (`Conversation.docx` or `dcmachinebook.pdf`)  
2. Build vocabulary  
3. Train a GPT model  
4. Save the trained model as **`wrd_trained_model.pkl`**  

📌 Early Stopping is included → if loss does not improve for 5 epochs (`min_delta=0.001`), training stops and the **best model** is saved.  

---

## ✨ Text Generation  

Text is generated using the `Solution.generate()` function with parameters:  

- `context` → The seed text (e.g., `"dc machine is"`)  
- `new_chars` → How many characters/wprds to generate
- `context_length` → How many tokens the model can look back  
- `temperature` → Controls creativity (higher = more random)  
- `top_k` → Sample only from top-k most likely words  

Example (word-level):  

```python
seed = "dc machine is" # this will be the starting point of model
encoded_seed = [stoi[w] for w in seed.split() if w in stoi]
start_context = torch.tensor([encoded_seed], dtype=torch.long).to(device)

generated_answer = solution.generate(
    model=model,
    new_chars=100,
    context=start_context,
    context_length=block_size,
    int_to_char=itos,
    temperature=0.8,
    top_k=20,
    join_with=" "
)

print("Q:", seed)
print("A:", generated_answer)
```

---

## 📊 Results  

- **Character-level GPT** → training attempted, but no usable results saved.  
- **Word-level GPT** → trained on **~17,000 characters (~3,500 words) until Chapter 1 of DC Machines book**.  
- Trained model saved as **`wrd_trained_model.pkl`** and can be reused for inference.  

Screenshots of sample outputs are available in `Word_level/outputs/`.  

---

## 🔗 Clone the Repository  

```bash
git clone https://github.com/moabs-dev/GPT-implementation_from_scratch
```

---

## 📜 Notes  

- This repo is **educational only** — showing how GPT works step by step.  
- No frontend/backend provided.  
- You can experiment with your own `.docx` or `.pdf` datasets by placing them inside the repo and updating the training scripts.  

---

✨ Enjoy exploring how GPT works under the hood! 
Explanation of how attention works in Gpt class is explained in folder `explaining`
