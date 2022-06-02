def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0,1,2,4,3,5).contiguous().view(-1, C, window_size, window_size)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    eps = 10e-10   #偶尔出现了B=1的情形
    print((windows.shape[0]) / (H * W / window_size / window_size))
    B = int((windows.shape[0]) / (H * W / window_size / window_size))
    x = windows.view(B, -1, H // window_size, W // window_size, window_size, window_size)
    x = x.permute(0,1,2,4,3,5).contiguous().view(B, -1, H, W)
    return x

def combine(patches, patch_size, lap):
    patches_row = combine_son(patches, patch_size, lap)
    patches_col = combine_son(patches_row, patch_size, lap)
    return patches_col

def combine_son(patches, patch_size, lap):
    n_w = patches.shape[1]
    num_lap = int(np.ceil(n_w // patch_size - 1))

    sigmoid = lambda x: 1 / (1 + np.exp(-x + 1e-15))
    weight1 = np.diag(sigmoid(np.linspace(-3, 3, lap)))

    weight2 = np.eye(np.shape(weight1)[0]) - weight1
    weight = np.concatenate([weight1, weight2], axis=0)
    bound_loc = [patch_size * (item + 1) for item in list(range(num_lap))]

    result = copy.deepcopy(patches)
    print(bound_loc)
    for idx in bound_loc:
        result[:, idx - lap:idx] = np.dot(patches[:, idx - lap:idx + lap], weight)
    count = 0
    for idx in reversed(bound_loc):
        count = count + 1
        result = np.delete(result, list(range(idx, idx + lap)), 1)
    return result


def Patch(img, Windows_size, Overlap_size):

    Step_size = Windows_size - Overlap_size
    Height, Width = img.shape
    Width_new = ((Width-Windows_size)//Step_size+1)*Step_size+Windows_size
    Height_new = ((Height-Windows_size)// Step_size + 1) * Step_size+Windows_size
    img = np.pad(img, ((0, Height_new - img.shape[0]), (0, Width_new - img.shape[1])), 'symmetric')
    img_lapcol = Overlap(img,Windows_size,Overlap_size)
    img_laprow = Overlap(img_lapcol.T,Windows_size,Overlap_size)
    return img_laprow

def Overlap(img,Windows_size,Overlap_size):
    Step_size = Windows_size - Overlap_size
    Height, Width = img.shape
    print(img.max())
    num_windows = ((Width - Windows_size) // Step_size + 1)
    N_col = Windows_size * num_windows
    Matrix_extend = np.zeros((Width,N_col))
    Matrix = np.eye(Width)
    for i in range(num_windows-1,-1,-1):
        Matrix_extend[:,i*Windows_size:(i+1)*Windows_size] = Matrix[:,i*Step_size:i*Step_size+Windows_size]
    img_lap = np.dot(img, Matrix_extend)
    print(img_lap.max())
    return img_lap

def Anti_Normlize(img,result):
    B, C, H, W = img.shape

    temp = img.view(img.size(0), -1)
    result = result.view(result.size(0), -1)

    result *= (temp.max(1, keepdim=True)[0]-temp.min(1, keepdim=True)[0])
    result += temp.min(1, keepdim=True)[0]

    result = result.view(B, C, H, W)
    noise = img - result
    return result,noise

def Anti_Overlap(img,Windows_size,Overlap_size):
    B, C, H, W = img.shape
    img_out = np.zeros(600)
    Step_size = Windows_size - Overlap_size
    Block=[]
    Num = W // Windows_size
    #不求矩阵直接做
    for i in range(Num):
        if i == 0:
            Block.append(img[:, Step_size])
        elif i == Num-1:
            Block.append(0.5*img[:,Windows_size*i-Overlap_size:Windows_size*i]+0.5*img[:,Windows_size*i:Windows_size*i+Overlap_size])
            Block.append(img[:, Windows_size * i + Overlap_size:])
        else:
            Block.append(0.5 * img[:, Windows_size * i - Overlap_size:Windows_size * i]+0.5*img[:,Windows_size*i:Windows_size*i+Overlap])
            Block.append(img[:, Windows_size * i + Overlap_size:Windows_size * (i+1) - Overlap_size])
    return np.array(Block)