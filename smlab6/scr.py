import numpy as np
import sys
from tqdm import tqdm
import cv2 

def get_size(obj, seen=None):
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, np.ndarray):
        size = obj.nbytes
    elif isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def rle_encode(data):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    shape_info = np.array([len(data.shape)])
    shape_info = np.concatenate([shape_info, data.shape])
    
    flattened = data.flatten()
    
    buffer = np.zeros(np.prod(data.shape) * 2, dtype=flattened.dtype)
    buffer_pos = 0
    
    if len(flattened) == 0:
        return np.concatenate([shape_info, np.array([])])
    
    current_val = flattened[0]
    count = 1
    
    for i in tqdm(range(1, len(flattened)), desc="RLE Encoding"):
        if flattened[i] == current_val:
            count += 1
        else:
            buffer[buffer_pos] = count
            buffer[buffer_pos+1] = current_val
            buffer_pos += 2
            current_val = flattened[i]
            count = 1

    buffer[buffer_pos] = count
    buffer[buffer_pos+1] = current_val
    buffer_pos += 2
    
    compressed = buffer[:buffer_pos]
    
    return np.concatenate([shape_info, compressed])

def rle_decode(compressed):
    dims = int(compressed[0])
    shape = tuple(compressed[1:dims+1].astype(int))
    data = compressed[dims+1:]
    
    decoded = np.zeros(np.prod(shape), dtype=data.dtype)
    decoded_pos = 0
    
    for i in tqdm(range(0, len(data), 2), desc="RLE Decoding"):
        count = data[i]
        value = data[i+1]
        decoded[decoded_pos:decoded_pos+count] = value
        decoded_pos += count
    
    return decoded.reshape(shape)

def find_repeated(data, start):
    if start >= len(data) - 1:
        return 1
    value = data[start]
    length = 1
    while start + length < len(data) and data[start + length] == value and length < 127:
        length += 1
    return length

def find_different(data, start):
    if start >= len(data) - 1:
        return 1
    length = 1
    while (start + length < len(data) - 1 and
           data[start + length] != data[start + length + 1] and
           length < 127):
        length += 1
    if start + length == len(data) - 1:
        length += 1
    return min(length, 127)

def byterun_encode(data):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    shape_info = np.array([len(data.shape)])
    shape_info = np.concatenate([shape_info, data.shape])
    
    flattened = data.flatten()
    
    buffer = np.zeros(np.prod(data.shape) * 2, dtype=flattened.dtype)
    buffer_pos = 0
    pos = 0
    
    if len(flattened) == 0:
        return np.concatenate([shape_info, np.array([])])
    
    with tqdm(total=len(flattened), desc="ByteRun Encoding") as pbar:
        while pos < len(flattened):
            if flattened[pos] == flattened[pos + 1] if pos < len(flattened) - 1 else False:
                repeated_len = find_repeated(flattened, pos)
                while repeated_len > 0:
                    chunk = min(repeated_len, 128)
                    buffer[buffer_pos] = -chunk + 1
                    buffer[buffer_pos+1] = flattened[pos]
                    buffer_pos += 2
                    pos += chunk
                    repeated_len -= chunk
                    pbar.update(chunk)
            else:
                different_len = find_different(flattened, pos)
                chunk = min(different_len, 128)
                buffer[buffer_pos] = chunk - 1
                buffer_pos += 1
                buffer[buffer_pos:buffer_pos+chunk] = flattened[pos:pos+chunk]
                buffer_pos += chunk
                pos += chunk
                pbar.update(chunk)
    
    compressed = buffer[:buffer_pos]
    
    return np.concatenate([shape_info, compressed])

def byterun_decode(compressed):
    dims = int(compressed[0])
    shape = tuple(compressed[1:dims+1].astype(int))
    data = compressed[dims+1:]
    
    decoded = np.zeros(np.prod(shape), dtype=data.dtype)
    decoded_pos = 0
    data_pos = 0
    
    with tqdm(total=np.prod(shape), desc="ByteRun Decoding") as pbar:
        while decoded_pos < np.prod(shape) and data_pos < len(data):
            control = data[data_pos]
            data_pos += 1
            
            if control >= 0: 
                length = control + 1
                decoded[decoded_pos:decoded_pos+length] = data[data_pos:data_pos+length]
                decoded_pos += length
                data_pos += length
                pbar.update(length)
            else:  
                length = -control + 1
                value = data[data_pos]
                decoded[decoded_pos:decoded_pos+length] = value
                decoded_pos += length
                data_pos += 1
                pbar.update(length)
    
    return decoded.reshape(shape)

def test_compression(data, encoder, decoder, name):
    data = data.astype(int)
    
    print(f"\n=== Testing {name} with data shape {data.shape} ===")
    
    compressed = encoder(data)
    original_size = get_size(data)
    compressed_size = get_size(compressed)
    
    decompressed = decoder(compressed)
    
    assert np.array_equal(data, decompressed), "Decompressed data doesn't match original!"
    print("Success: decompressed data matches original")
    
    cr = original_size / compressed_size
    pr = (compressed_size / original_size) * 100
    
    print(f"Original size: {original_size} bytes")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Compression ratio (CR): {cr:.4f}")
    print(f"Percentage of original (PR): {pr:.2f}%")
    
    return cr, pr

def load_image_as_numpy(filename):

    try:
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)  # Загрузка с поддержкой альфа-канала
        if img is None:
            print(f"Error: cant download emage {filename}.")
            return None
        return img
    except cv2.error as e:
        print(f"Error while read {filename}: {e}")
        return None
    except FileNotFoundError:
        print(f"Error: File {filename} not find.")
        return None

if __name__ == "__main__":
    # Test cases
    test_cases = [
        np.array([1,1,1,1,2,1,1,1,1,2,1,1,1,1]),
        np.array([1,2,3,1,2,3,1,2,3]),
        np.array([5,1,5,1,5,5,1,1,5,5,1,1,5]),
        np.array([-1,-1,-1,-5,-5,-3,-4,-2,1,2,2,1]),
        np.zeros((1,520)),
        np.arange(0,521,1),
        np.eye(7),
        np.dstack([np.eye(7),np.eye(7), np.eye(7)]),
        np.ones((1,1,1,1,1,1,10))
    ]
    
    image_files = ["col.jpg", "rys.png", "wzor.png"]
    images_data = []

    for file in image_files:
        img_array = load_image_as_numpy(file)
        if img_array is not None:
            images_data.append(img_array)

    # Run tests
    for i, test_data in enumerate(test_cases):
        print(f"\n=== Test Case {i+1} ===")
        test_compression(test_data, rle_encode, rle_decode, "RLE")
        test_compression(test_data, byterun_encode, byterun_decode, "ByteRun")
    
    # Test compression on images
    if images_data:
        print("\n=== Image Compression Tests ===")
        for i, img_data in enumerate(images_data):
            print(f"\n--- {image_files[i]} ---")
            test_compression(img_data, rle_encode, rle_decode, "RLE")
            test_compression(img_data, byterun_encode, byterun_decode, "ByteRun")