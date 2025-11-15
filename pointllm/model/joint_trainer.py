import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

class JointFeatureAlignmentTrainer:
    def __init__(self, teacher_model, student_model, train_dataset, val_dataset=None,
                 lr=1e-4, save_dir="./output", device=None, use_cosine=True):
        self.teacher = teacher_model.eval()
        self.student = student_model.train()
        self.device = device or torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu")
        self.teacher.to(self.device)
        self.student.to(self.device)
        self.train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=2) if val_dataset else None
        self.opt = torch.optim.Adam(self.student.parameters(), lr=lr)
        self.save_dir = save_dir
        self.use_cosine = use_cosine

    def feature_loss(self, f_s, f_t):
        loss = F.mse_loss(f_s, f_t)
        if self.use_cosine:
            cos = 1 - F.cosine_similarity(f_s, f_t, dim=-1).mean()
            loss += 0.1 * cos
        return loss

    def train(self, num_epochs=20, save_interval=5):
        for epoch in range(num_epochs):
            self.student.train()
            total_loss = 0
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                orig_pts = batch["original"].to(self.device)
                grasp_pts = batch["grasp"].to(self.device)
                with torch.no_grad():
                    f_t = self.teacher(orig_pts)
                f_s = self.student(grasp_pts)
                loss = self.feature_loss(f_s, f_t)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}: avg loss={total_loss/len(self.train_loader):.5f}")
            if (epoch + 1) % save_interval == 0:
                torch.save(self.student.state_dict(),
                           f"{self.save_dir}/adapter_epoch{epoch+1}.pth")
