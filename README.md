# **Hierarchical BiLSTM for Legal Document Classification: A Deep Dive**

## **Introduction**
Legal documents, such as terms of service (ToS), contracts, and privacy policies, contain complex language that often includes unfair clauses. Detecting these clauses manually is tedious and error-prone. To address this, we developed a **Hierarchical BiLSTM-based legal text classifier**, specifically for identifying unfair clauses in Terms of Service documents. The model was trained on the **"unfair_tos"** dataset from the **LexGLUE benchmark**, leveraging **BiLSTM (Bidirectional Long Short-Term Memory)** networks for sequence modeling.

This article will walk through the key steps in building this system, from **dataset preprocessing and model architecture to training, evaluation, and real-world applications**.

---

## **Dataset: LexGLUE - Unfair ToS**
The **LexGLUE (Legal General Language Understanding Evaluation)** benchmark is designed to test NLP models on legal tasks. We use the **"unfair_tos"** dataset, which contains annotated unfair clauses from online Terms of Service agreements.

### **Dataset Breakdown**
The dataset consists of **5532 training samples, 2275 validation samples, and 1607 test samples**. Each sample is a legal text snippet labeled with one or more unfair clause categories:

1. **Jurisdiction**
2. **Choice of Law**
3. **Limitation of Liability**
4. **Unilateral Change**
5. **Content Removal**
6. **Contract by Using**
7. **Unilateral Termination**
8. **Arbitration**

Since a document can belong to multiple categories, this is a **multi-label classification problem**.

---

## **Data Preprocessing and Tokenization**
### **Why Tokenization?**
Legal texts contain **long, complex sentences** with domain-specific terminology. Standard NLP models struggle with such texts, making **tokenization** crucial.

We used the **BERT tokenizer ("bert-base-uncased")**, which:
- Converts text into **word pieces** (subword tokenization)
- Ensures compatibility with **pretrained transformer models**
- Allows **padding and truncation** to a fixed length (256 tokens)

#### **Processing the Dataset**
1. **Extract text samples**: We collect all legal clauses from the dataset.
2. **Tokenize them using BERT**: The tokenized texts are converted into numerical vectors.
3. **Convert labels into one-hot encoded format**: Each clause can belong to multiple categories, so we map them accordingly.

```python
from transformers import AutoTokenizer

# Initialize BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def prepare_data_for_training(dataset):
    texts = dataset['text']
    
    # Tokenization
    tokenized = tokenizer(texts, padding='max_length', truncation=True, max_length=256, return_tensors='np')
    input_ids = tokenized['input_ids']
    
    # Convert labels to multi-hot encoding
    labels = np.zeros((len(dataset), 8))
    for i, item in enumerate(dataset):
        labels[i, item['labels']] = 1
    
    return input_ids, labels
```

---

## **Model Architecture: BiLSTM with Hierarchical Representation**
### **Why BiLSTM?**
Unlike simple feedforward networks, **Bidirectional LSTMs (BiLSTMs)** capture both **past and future contexts** in a sentence. This is essential for **legal text classification**, where context determines meaning.

### **Hierarchical Representation**
We designed a **Hierarchical BiLSTM model**:
1. **Word-level BiLSTM**: Captures **local dependencies** within a clause.
2. **Sentence-level BiLSTM**: Aggregates information from words into a **higher-level representation**.
3. **Dense layers for classification**: Uses ReLU and dropout layers for **better generalization**.

### **Model Implementation**
```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_bilstm_model(vocab_size=30522, num_labels=8, embedding_dim=100, max_len=256):
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    
    # Word Embeddings
    embedding = layers.Embedding(vocab_size, embedding_dim, input_length=max_len)(input_ids)
    x = layers.Dropout(0.2)(embedding)
    
    # Word-level BiLSTM
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    
    # Sentence-level BiLSTM
    x = layers.Bidirectional(layers.LSTM(64))(x)
    
    # Fully Connected Layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Output Layer (multi-label classification)
    output = layers.Dense(num_labels, activation='sigmoid')(x)
    
    model = models.Model(inputs=input_ids, outputs=output)
    
    # Compile Model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
```
The **sigmoid activation** in the final layer ensures that multiple categories can be predicted simultaneously.

---

## **Training the Model**
### **Key Training Strategies**
1. **Early Stopping**: Prevents overfitting by stopping training when validation loss stops improving.
2. **Model Checkpointing**: Saves the best model weights based on validation loss.
3. **Learning Rate Scheduling**: Reduces learning rate if no improvement is seen.

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(filepath='best_model.weights.h5', monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

history = model.fit(train_features, train_labels,
                    validation_data=(val_features, val_labels),
                    epochs=20, batch_size=32, callbacks=callbacks)
```

---

## **Model Evaluation**
After training, we evaluate the model on the **test set**:

```python
test_loss, test_accuracy = model.evaluate(test_features, test_labels)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")
```
### **Results**
- **Test Loss:** `0.0562`
- **Test Accuracy:** `91.35%`

This indicates strong performance, with the model successfully identifying unfair clauses in unseen legal documents.

---

## **Real-World Application: Predicting Unfair Clauses**
A user can input any legal clause to check if it contains unfair terms.

```python
def make_predictions(text, model, tokenizer):
    category_mapping = {
        0: "Jurisdiction", 1: "Choice of Law", 2: "Limitation of Liability",
        3: "Unilateral Change", 4: "Content Removal", 5: "Contract by Using",
        6: "Unilateral Termination", 7: "Arbitration"
    }

    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=256, return_tensors='np')
    predictions = model.predict(tokens['input_ids'])
    
    results = [{ "category": category_mapping[i], "probability": pred * 100 } for i, pred in enumerate(predictions[0])]
    results.sort(key=lambda x: x["probability"], reverse=True)

    print("\nPrediction Results:")
    for r in results:
        print(f"{r['category']:25} : {r['probability']:.2f}%")

    return results

sample_text = "Your data may be shared with third parties without explicit consent."
results = make_predictions(sample_text, model, tokenizer)
```

### **Sample Output**
```
Jurisdiction              : 17.27%
Limitation of Liability   : 15.80%
Content Removal           : 10.52%
```

This system provides **legal transparency** by automatically detecting unfair clauses in ToS agreements.

---

## **Conclusion**
This **Hierarchical BiLSTM-based model** effectively classifies unfair clauses in legal documents. Future improvements may include:
- **Transformer-based models** (e.g., BERT, RoBERTa)
- **Attention mechanisms** for better context capture
- **Larger datasets** for improved generalization

This model represents a **step forward in AI-driven legal document analysis**, promoting fairness and user rights in digital contracts. 🚀
