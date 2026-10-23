"""DAG-UNet implementation used in the experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DAG_UNet"]




def cosine_similarity_matrix(feature_map,threshold):

    # Get the shape of the feature map
    n, c, h, w = feature_map.shape
    
    # Reshape the feature map to (n, c, h*w), flattening the spatial dimensions
    flattened = feature_map.view(n, c, h * w)  # Shape: (n, c, h*w)
    
    # Normalize each channel vector
    flattened_norm = F.normalize(flattened, dim=-1)  # Normalize along the h*w dimension
    
    # Compute cosine similarity: (n, c, h*w) @ (n, h*w, c) -> (n, c, c)
    cosine_sim = torch.matmul(flattened_norm, flattened_norm.transpose(1, 2))  # Shape: (n, c, c)
    
    # Extract the upper triangle (excluding diagonal)
    # Set diagonal=1 to exclude diagonal; diagonal=0 will include diagonal
    upper_triangle = torch.triu(cosine_sim, diagonal=1)


    # Create a mask where values less than the threshold are set to 1, and others to 0
    mask = (upper_triangle < threshold).float()  # Shape: (n, c, c)
    
    # Sum along the row axis (dim=1) to count how many values in each column are < threshold
    count_smaller = mask.sum(dim=1)  # Shape: (n, c)

    
    # sum_per_batch = count_smaller.sum(dim=1)
    

    return count_smaller,upper_triangle

def select_top_channels(x, images, fraction):

    batch_size, num_channels, height, width = images.shape
    # Calculate how many channels to select based on the fraction
    num_select = int(fraction * num_channels)

    # Get the indices of the top 'num_select' values in x for each batch
    top_indices = torch.topk(x, num_select, dim=1, largest=True, sorted=False).indices  # Shape: (batch_size, num_select)

    # Create a mask for all indices of the channels
    all_indices = torch.arange(num_channels, device=images.device).unsqueeze(0).expand(batch_size, -1)  # Shape: (batch_size, num_channels)
    
    # Create a mask to identify the selected channels
    selected_mask = torch.zeros_like(all_indices, dtype=torch.bool)
    selected_mask.scatter_(1, top_indices, True)  # Mark the top indices as True

    # Use the mask to select and unselect channels efficiently
    selected_channels = images[selected_mask].view(batch_size, num_select, height, width)
    unselected_channels = images[~selected_mask].view(batch_size, num_channels - num_select, height, width)

    return selected_channels, unselected_channels



class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation = 1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups = dim, dilation = dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups = dim, dilation = dilation)

    def forward(self, x):
        x = x + self.dw_h(x) + self.dw_w(x)
        return x

class EncoderBlock(nn.Module):
    """Encoding then downsampling"""
    def __init__(self, in_c, out_c, mixer_kernel = (3, 3),threshold=0.5,frac=0.75):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel = (3, 3))
        self.bn = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(int(in_c*frac), int(out_c-in_c*(1-frac)), kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.frac = frac
        self.down = nn.MaxPool2d((2,2))
        self.act = nn.GELU()
        self.threshold = threshold
    def forward(self, x):
        skip = self.bn(self.dw(x))

        x_similar,upper = cosine_similarity_matrix(skip,threshold=self.threshold)
        x_selected,x_unselected = select_top_channels(x_similar, skip, fraction=self.frac) #16

        x = self.pw(x_selected)

        x_pconv = torch.cat((x,x_unselected),dim=1)
        x_pconv = self.bn1(x_pconv)
        x = self.act(self.down(x_pconv))

        return x, skip,upper



class FGlo(nn.Module):
    """
    the FGlo class is employed to refine the joint feature of both local feature and surrounding context.
    """
    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y



class EncoderBlock1(nn.Module):
    """Encoding then downsampling"""
    def __init__(self, in_c, out_c, mixer_kernel = (7, 7),threshold=0.5,frac=0.75):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel = (7, 7))
        self.bn = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(int(in_c*frac), int(out_c-in_c*(1-frac)), kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.frac = frac
        self.down = nn.MaxPool2d((2,2))
        self.act = nn.GELU()
        self.threshold = threshold
        self.FGlo = FGlo(out_c)

        
        self.act = nn.PReLU(out_c)
    def forward(self, x):
        skip = self.bn(self.dw(x))

        x_similar,upper = cosine_similarity_matrix(skip,threshold=self.threshold)
        x_selected,x_unselected = select_top_channels(x_similar, skip, fraction=self.frac) #16

        x = self.pw(x_selected)

        x_pconv = torch.cat((x,x_unselected),dim=1)
        x_pconv = self.act(self.bn1(x_pconv))
        x_pconv = self.FGlo(x_pconv)
        x = self.act(self.down(x_pconv))

        return x, skip,upper

class EncoderBlock2(nn.Module):
    """Encoding then downsampling"""
    def __init__(self, in_c, out_c, mixer_kernel = (7, 7),threshold=0.5,frac=0.75):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel = (7, 7))
        self.bn = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(int(in_c*frac), int(out_c-in_c*(1-frac)), kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.frac = frac
        self.down = nn.MaxPool2d((2,2))
        self.act = nn.GELU()
        self.threshold = threshold
        self.FGlo = FGlo(out_c)

        
        self.act = nn.PReLU(out_c)
    def forward(self, x):
        skip = self.bn(self.dw(x))

        x_similar,upper = cosine_similarity_matrix(skip,threshold=self.threshold)
        x_selected,x_unselected = select_top_channels(x_similar, skip, fraction=self.frac) #16

        x = self.pw(x_selected)

        x_pconv = torch.cat((x,x_unselected),dim=1)
        x_pconv = self.act(self.bn1(x_pconv))
        x_pconv = self.FGlo(x_pconv)
        x = self.act(x_pconv)

        return x


class DecoderBlock(nn.Module):
    """Upsampling then decoding"""
    def __init__(self, in_c, out_c, mixer_kernel = (7, 7)):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pw = nn.Conv2d(in_c + out_c, out_c,kernel_size=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.dw = AxialDW(out_c, mixer_kernel = (7, 7))
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.pw2(self.dw(self.bn(self.pw(x)))))
        return x
    
class BottleNeckBlock(nn.Module):
    """Axial dilated DW convolution"""
    def __init__(self, dim):
        super().__init__()

        gc = dim//4
        self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)
        self.dw1 = AxialDW(gc, mixer_kernel = (3, 3), dilation = 1)
        self.dw2 = AxialDW(gc, mixer_kernel = (3, 3), dilation = 2)
        self.dw3 = AxialDW(gc, mixer_kernel = (3, 3), dilation = 3)

        self.bn = nn.BatchNorm2d(4*gc)
        self.pw2 = nn.Conv2d(4*gc, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.pw1(x)
        x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], 1)
        x = self.act(self.pw2(self.bn(x)))
        return x




class DAG_UNet(nn.Module):
    def __init__(self, in_c,out_c,threshold,frac):
        super().__init__()

        """Encoder"""
        self.conv_in = nn.Conv2d(in_c, 16, kernel_size=7, padding='same')
        self.e1 = EncoderBlock1(16, 32,threshold=threshold,frac=frac)
        self.e2 = EncoderBlock1(32, 64,threshold=threshold,frac=frac)
        self.e3 = EncoderBlock1(64, 128,threshold=threshold,frac=frac)
        self.e4 = EncoderBlock1(128, 256,threshold=threshold,frac=frac)
        self.e5 = EncoderBlock1(256, 512,threshold=threshold,frac=frac)

        """Bottle Neck"""
        self.b5 = EncoderBlock2(512,512)

        """Decoder"""
        self.d5 = DecoderBlock(512, 256)
        self.d4 = DecoderBlock(256, 128)
        self.d3 = DecoderBlock(128, 64)
        self.d2 = DecoderBlock(64, 32)
        self.d1 = DecoderBlock(32, 16)
        self.conv_out = nn.Conv2d(16, out_c, kernel_size=1)

    def forward(self, x):
        """Encoder"""
        x = self.conv_in(x)
        x, skip1,x_similar = self.e1(x)
        x, skip2,x_similar1  = self.e2(x)
        x, skip3,x_similar2 = self.e3(x)
        x, skip4,x_similar3 = self.e4(x)
        x, skip5,x_similar4 = self.e5(x)

        """BottleNeck"""
        x = self.b5(x)         # (512, 8, 8)

        """Decoder"""
        x = self.d5(x, skip5)
        x = self.d4(x, skip4)
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)
        x = self.conv_out(x)
        return x
