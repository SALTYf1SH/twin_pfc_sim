
import torch
import torch.nn as nn

class EvolutionLoss(nn.Module):
    def __init__(self): super(EvolutionLoss, self).__init__(); self.mse = nn.MSELoss()
    def forward(self, pred_t, target_t, target_prev):
        true_delta = target_t - target_prev
        pred_delta = pred_t - target_prev
        return self.mse(pred_delta, true_delta)

def test_redundancy():
    bs = 10
    pred = torch.randn(bs, 1, 64, 64)
    target = torch.randn(bs, 1, 64, 64)
    
    # Case 1: Random prev
    prev1 = torch.randn(bs, 1, 64, 64)
    
    # Case 2: Zero prev (Simulate the bug)
    prev2 = target.clone() 
    
    # Case 3: Proper prev
    prev3 = target - 0.1 # Small delta
    
    criterion = EvolutionLoss()
    mse = nn.MSELoss()
    
    loss1 = criterion(pred, target, prev1)
    loss2 = criterion(pred, target, prev2)
    loss3 = criterion(pred, target, prev3)
    loss_mse = mse(pred, target)
    
    print(f"MSE Loss: {loss_mse.item():.6f}")
    print(f"Evo Loss (Random Prev): {loss1.item():.6f}")
    print(f"Evo Loss (Zero Delta):  {loss2.item():.6f}")
    print(f"Evo Loss (Small Delta): {loss3.item():.6f}")
    
    if torch.allclose(loss1, loss_mse) and torch.allclose(loss2, loss_mse) and torch.allclose(loss3, loss_mse):
        print("CONFIRMED: EvolutionLoss is mathematically identical to MSELoss.")
    else:
        print("DISPROVED: EvolutionLoss is different.")

if __name__ == "__main__":
    test_redundancy()
