import torch.nn as nn
import pytorch_lightning as pl
import torch

class LSTMModel(pl.LightningModule):

    def __init__(self, hidden_size, lstm_layers, input_size=8, output_size=3, dropout=0.05):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=lstm_layers, dropout=dropout)
        
        self.out_linear = nn.Linear(hidden_size, output_size)
        
        # keep track of losses function.
        self.train_losses = []
        self.test_losses = []
        self.loss_func = nn.L1Loss()
        
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:,-1:,:]
        
        output = self.out_linear(lstm_out)
        return output


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y, y_hat)#.mean()
        self.train_losses.append(loss)
        return loss

    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y, y_hat)#.mean()
        self.test_losses.append(loss)
        return loss
    
    
    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_losses).mean()
        print(f'Test Loss: {avg_loss}')
        return {'L1_loss': avg_loss}
    
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_losses).mean()
        print(f'Train Loss: {avg_loss}')
        return {'L1_loss': avg_loss}
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


# Initialize the model and trainer
model = LSTMModel(output_size=1, hidden_size=128, lstm_layers=6, dropout=0.0)

checkpoint = torch.load("model\LSTM_BTC\checkpoints\epoch=99-step=900.ckpt")
model.load_state_dict(checkpoint['state_dict'])