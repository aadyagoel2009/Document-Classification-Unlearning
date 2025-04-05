import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS
import snntorch as snn

# Dataset Class for AG News
class AGNewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer, max_length=500):
        self.texts = texts
        # Convert labels from 1-indexed to 0-indexed.
        self.labels = torch.tensor([label - 1 for label in labels], dtype=torch.long)
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_length = max_length

    def text_pipeline(self, text):
        tokenized = self.tokenizer(text)
        # With default index set, this simply converts each token
        indexed = [self.vocab[token] for token in tokenized]
        padded = indexed[:self.max_length] + [0] * (self.max_length - len(indexed))
        return torch.tensor(padded, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.text_pipeline(self.texts[idx]), self.labels[idx]

# DataLoader for AG News
class DataLoaderAGNews:
    def __init__(self, batch_size=32, max_length=500):
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = get_tokenizer("basic_english")
        
        # Load AG_NEWS splits (each sample is a tuple (label, text))
        train_iter = AG_NEWS(split='train')
        test_iter = AG_NEWS(split='test')
        self.train_data = list(train_iter)
        self.test_data = list(test_iter)
        
        # Unpack texts and labels
        self.train_texts = [text for (label, text) in self.train_data]
        self.train_labels = [label for (label, text) in self.train_data]
        self.test_texts = [text for (label, text) in self.test_data]
        self.test_labels = [label for (label, text) in self.test_data]
        
        # AG News has 4 classes
        self.num_classes = 4
        
        self.vocab = self.build_vocab(self.train_texts)
        self.vocab.set_default_index(self.vocab["<unk>"])
        
        self.train_dataset = AGNewsDataset(self.train_texts, self.train_labels, self.vocab, self.tokenizer, max_length)
        self.test_dataset = AGNewsDataset(self.test_texts, self.test_labels, self.vocab, self.tokenizer, max_length)

    def build_vocab(self, texts):
        def yield_tokens(data_iter):
            for text in data_iter:
                yield self.tokenizer(text)
        return build_vocab_from_iterator(yield_tokens(texts), specials=["<unk>"])

    def get_dataloader(self, split="train"):
        dataset = self.train_dataset if split == "train" else self.test_dataset
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

# TextCNN Model as in the paper
class TextCNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=[3, 4, 5], num_filters=100):
        super(TextCNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Create a convolution for each kernel size
        self.conv_layers = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        x = self.embedding(x)           # (batch_size, seq_length, embed_dim)
        x = x.permute(0, 2, 1)            # (batch_size, embed_dim, seq_length)
        conv_results = []
        for conv in self.conv_layers:
            out = conv(x)
            # If the conv output comes as a tuple (after SNN conversion), use the first element
            if isinstance(out, tuple):
                out = out[0].detach()
            else:
                out = torch.relu(out)
            # Max-pool over the temporal dimension
            conv_results.append(out.max(dim=2)[0])
        x = torch.cat(conv_results, dim=1)
        x = self.dropout(x)
        return self.fc(x)

# Helper function to safely reset spiking states
def reset_spiking_states(model):
    for module in model.modules():
        if hasattr(module, "reset") and callable(module.reset):
            module.reset()

# SNN Conversion Module
class SNNConverter:
    def __init__(self, textcnn_model, num_classes):
        self.textcnn = textcnn_model
        self.snn = self.create_snn(num_classes)

    def create_snn(self, num_classes):
        # Create a new model and load CNN weights
        snn_model = TextCNNModel(len(self.textcnn.embedding.weight), 128, num_classes)
        snn_model.load_state_dict(torch.load("textcnn_agnews.pth"))
        self.normalize_weights(snn_model)
        
        # Replace ReLU with spiking (Leaky LIF) layers for each conv layer.
        for i in range(len(snn_model.conv_layers)):
            snn_model.conv_layers[i] = nn.Sequential(
                snn_model.conv_layers[i],  # Original conv layer
                snn.Leaky(beta=0.95, threshold=1.0, learn_beta=False,
                          spike_grad=snn.surrogate.fast_sigmoid(slope=25))  # SNN layer mimicking paper
            )
        return snn_model

    def normalize_weights(self, snn_model):
        """Normalize CNN weights before conversion to an SNN as per the paper."""
        with torch.no_grad():
            for i in range(len(self.textcnn.conv_layers)):
                conv_layer = self.textcnn.conv_layers[i]
                max_act = torch.max(torch.abs(conv_layer.weight))
                if max_act > 0:
                    snn_model.conv_layers[i].weight.data /= max_act
                    snn_model.conv_layers[i].bias.data /= max_act

    def fine_tune_snn(self, train_loader, lr=0.001, epochs=5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.snn.to(device)
        optimizer = torch.optim.Adam(self.snn.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.snn.train()
            total_loss, correct = 0, 0
            for texts, labels in train_loader:
                # Reset spiking states before each batch to avoid carrying over state
                reset_spiking_states(self.snn)
                texts, labels = texts.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.snn(texts)
                loss = criterion(outputs, labels)
                loss.backward()  # Make sure not to use retain_graph=True unless needed
                optimizer.step()
                total_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
            
            print(f"SNN Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {correct/len(train_loader.dataset):.4f}")
        
        torch.save(self.snn.state_dict(), "snn_agnews.pth")

    def evaluate_snn(self, test_loader):
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

# Main function: Train CNN, convert to SNN, and fine-tune SNN.
def main():
    batch_size = 32
    embed_dim = 128
    num_classes = 4  # AG News
    num_epochs = 5  # Adjust epochs as needed

    # Load AG News Data
    data_loader = DataLoaderAGNews(batch_size=batch_size)
    train_loader = data_loader.get_dataloader("train")
    test_loader = data_loader.get_dataloader("test")
    
    # Train TextCNN (ensure proper training to get high accuracy)
    textcnn = TextCNNModel(len(data_loader.vocab), embed_dim, num_classes)
    textcnn.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    textcnn.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(textcnn.parameters(), lr=0.001)
    
    print("Training CNN model...")
    for epoch in range(num_epochs):
        textcnn.train()
        total_loss, correct = 0, 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = textcnn(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
        print(f"CNN Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {correct/len(train_loader.dataset):.4f}")
    
    # Save the trained CNN weights.
    torch.save(textcnn.state_dict(), "textcnn_agnews.pth")
    
    # Convert CNN to SNN and fine-tune.
    snn_converter = SNNConverter(textcnn, num_classes)
    print("Fine-tuning SNN model...")
    snn_converter.fine_tune_snn(train_loader, lr=0.001, epochs=num_epochs)
    
    # Evaluate the SNN model.
    snn_converter.evaluate_snn(test_loader)

if __name__ == "__main__":
    main()
