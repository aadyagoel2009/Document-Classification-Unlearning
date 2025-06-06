import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import snntorch as snn

# Dataset Class for 20 Newsgroups
class NewsGroupDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer, max_length=500):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_length = max_length

    def text_pipeline(self, text):
        tokenized = self.tokenizer(text)
        # Option 1: Use your manual check
        indexed = [self.vocab[token] if token in self.vocab else self.vocab["<unk>"] for token in tokenized]
        # Option 2: (Recommended) If you set a default index, you can simply do:
        # indexed = [self.vocab[token] for token in tokenized]
        padded = indexed[:self.max_length] + [0] * (self.max_length - len(indexed))
        return torch.tensor(padded, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.text_pipeline(self.texts[idx]), self.labels[idx]

# DataLoader Class
class DataLoader20Newsgroups:
    def __init__(self, batch_size=32, max_length=500):
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = get_tokenizer("basic_english")
        
        newsgroups = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
        self.texts = newsgroups.data
        self.labels = newsgroups.target
        self.num_classes = len(newsgroups.target_names)
        
        self.train_texts, self.test_texts, self.train_labels, self.test_labels = train_test_split(
            self.texts, self.labels, test_size=0.2, random_state=42)
        
        self.vocab = self.build_vocab()
        # Optionally, set the default index for unknown tokens:
        self.vocab.set_default_index(self.vocab["<unk>"])
        
        self.train_dataset = NewsGroupDataset(self.train_texts, self.train_labels, self.vocab, self.tokenizer, max_length)
        self.test_dataset = NewsGroupDataset(self.test_texts, self.test_labels, self.vocab, self.tokenizer, max_length)

    def build_vocab(self):
        def yield_tokens(data_iter):
            for text in data_iter:
                yield self.tokenizer(text)
        return build_vocab_from_iterator(yield_tokens(self.train_texts), specials=["<unk>"])

    def get_dataloader(self, split="train"):
        dataset = self.train_dataset if split == "train" else self.test_dataset
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

# TextCNN Model
class TextCNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=[3, 4, 5], num_filters=100):
        super(TextCNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv_layers = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # Shape: (batch_size, embed_dim, seq_length)
        conv_results = []
        for conv in self.conv_layers:
            out = conv(x)
            # If the output is a tuple (as it might be after SNN conversion), use the first element
            if isinstance(out, tuple):
                out = out[0].detach()
            else:
                out = torch.relu(out)
            # Max-pool over the temporal dimension (dim=2)
            conv_results.append(out.max(dim=2)[0])
        x = torch.cat(conv_results, dim=1)
        x = self.dropout(x)
        return self.fc(x)

# Convert to SNN
class SNNConverter:
    def __init__(self, textcnn_model, num_classes):
        self.textcnn = textcnn_model
        self.snn = self.create_snn(num_classes)

    def create_snn(self, num_classes):
        """Convert a trained CNN to an SNN by replacing ReLU with LIF neurons."""
        snn_model = TextCNNModel(len(self.textcnn.embedding.weight), 128, num_classes)
        snn_model.load_state_dict(torch.load("textcnn_20news.pth"))
        self.normalize_weights(snn_model)
        
        # Convert ReLU layers into LIF neurons
        for i in range(len(snn_model.conv_layers)):
            snn_model.conv_layers[i] = nn.Sequential(
                snn_model.conv_layers[i],  # Original Conv layer
                snn.Leaky(beta=0.95, threshold=1.0, learn_beta=False, spike_grad=snn.surrogate.fast_sigmoid(slope=25))  # SNN Layer
            )
        
        return snn_model

    def normalize_weights(self, snn_model):
        """Normalize CNN weights before conversion to an SNN."""
        with torch.no_grad():
            for i in range(len(self.textcnn.conv_layers)):
                max_act = torch.max(torch.abs(self.textcnn.conv_layers[i].weight))
                snn_model.conv_layers[i].weight.data /= max_act
                snn_model.conv_layers[i].bias.data /= max_act

    def fine_tune_snn(self, train_loader, lr=0.001, epochs=5):
        """Fine-tune the SNN after conversion using surrogate gradient learning."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.snn.to(device)
        optimizer = torch.optim.Adam(self.snn.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.snn.train()
            total_loss, correct = 0, 0
            for texts, labels in train_loader:
                texts, labels = texts.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.snn(texts)
                loss = criterion(outputs, labels)
                loss.backward()  # retain_graph=True is usually not needed here
                optimizer.step()
                total_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
            
            print(f"SNN Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {correct/len(train_loader.dataset):.4f}")
        
        torch.save(self.snn.state_dict(), "snn_20news.pth")

    def evaluate_snn(self, test_loader):
        """Evaluate the accuracy of the SNN on the test set."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.snn.to(device)
        self.snn.eval()
        
        correct, total = 0, 0
        with torch.no_grad():
            for texts, labels in test_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = self.snn(texts)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        print(f"SNN Test Accuracy: {correct / total:.4f}")

# Main Function
def main():
    batch_size, embed_dim, num_classes = 32, 128, 20
    
    # Load Data
    data_loader = DataLoader20Newsgroups(batch_size=batch_size)
    train_loader = data_loader.get_dataloader("train")
    test_loader = data_loader.get_dataloader("test")
    
    # Train TextCNN
    textcnn = TextCNNModel(len(data_loader.vocab), embed_dim, num_classes)
    textcnn.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    textcnn.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(textcnn.parameters(), lr=0.001)
    for epoch in range(5):
        textcnn.train()
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = textcnn(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    torch.save(textcnn.state_dict(), "textcnn_20news.pth")
    
    # Convert and Fine-tune SNN
    snn_converter = SNNConverter(textcnn, num_classes)
    snn_converter.fine_tune_snn(train_loader)
    snn_converter.evaluate_snn(test_loader)

if __name__ == "__main__":
    main()