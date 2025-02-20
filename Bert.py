from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model BERT yang sudah dilatih
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Contoh email
email_texts = ["Your account is locked! Click here to unlock: http://phishing.com"]

# Tokenisasi teks
inputs = tokenizer(email_texts, padding=True, truncation=True, return_tensors="pt")

# Prediksi
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=1)

# Hasil
label = "Phishing" if predictions[0] == 1 else "Legitimate"
print(f"🔍 Email: {email_texts[0]}\n🔮 Prediksi: {label}")

# load load bert

from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT model dan tokenizer
model_name = "bert-base-uncased"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Set model ke mode evaluasi
model.eval()
#Load fungsi deteksi phising 
def detect_phishing(email_text):
    # Tokenisasi teks email
    inputs = tokenizer(email_text, padding=True, truncation=True, return_tensors="pt")
    
    # Prediksi dengan model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Ambil hasil prediksi
    prediction = torch.argmax(outputs.logits, dim=1).item()
    
    # Konversi label menjadi teks
    label = "Phishing" if prediction == 1 else "Legitimate"
    
    return label
 #uji coba dengan cotoh email
 emails = [
    "Your account is locked! Click here to unlock: http://phishing.com",
    "Reminder: Your meeting is scheduled for today at 3 PM."
]

# Lakukan deteksi phishing
for email in emails:
    result = detect_phishing(email)
    print(f"🔍 Email: {email}\n🔮 Prediksi: {result}\n")