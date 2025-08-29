# LessWrong Contextual Chunking - Mac Setup Guide

## 🍎 Mac Hardware Requirements

### **Memory Requirements for BGE-M3 Embeddings:**

**Minimum Configuration:**
- **RAM:** 16GB (basic functionality)
- **Available RAM:** 12GB+ (with other apps closed)
- **Batch size:** 16-32 texts
- **Processing time:** ~8-12 hours

**Recommended Configuration:**  
- **RAM:** 32GB+ (optimal performance)
- **Available RAM:** 24GB+ 
- **Batch size:** 64-128 texts
- **Processing time:** ~4-6 hours

**High-Performance Configuration:**
- **RAM:** 64GB+ (enterprise/workstation)
- **Available RAM:** 48GB+
- **Batch size:** 256+ texts  
- **Processing time:** ~2-3 hours

### **Mac-Specific Optimizations:**

**Apple Silicon (M1/M2/M3) Macs:**
- ✅ **Native PyTorch MPS support** (Metal Performance Shaders)
- ✅ **Unified memory architecture** (CPU/GPU share RAM)
- ✅ **Better thermal management** than x86
- 🚀 **~2-3x faster** than Intel Macs

**Intel Macs:**
- ✅ **Standard CPU processing**  
- ⚠️ **Higher thermal throttling risk**
- ⚠️ **Slower than Apple Silicon**

## 🔧 Mac Setup Instructions

### 1. Environment Setup
```bash
# Create conda environment (recommended for Mac)
conda create -n lw_search python=3.11
conda activate lw_search

# Install dependencies with Mac optimizations
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers[all] faiss-cpu pandas numpy tqdm

# For Apple Silicon Macs (M1/M2/M3) - use MPS acceleration
pip install torch torchvision torchaudio  # Automatically includes MPS support
```

### 2. System Optimization
```bash
# Close unnecessary applications
# Monitor memory usage: Activity Monitor > Memory tab
# Ensure 12GB+ "Memory Available" before starting

# Optional: Increase swap space if needed
sudo sysctl -w vm.swapusage  # Check current swap
```

### 3. Choose Processing Strategy

**Option A: Single Process (Safest)**
```bash
python mac_optimized_embedder.py --mode single --batch-size 32
```

**Option B: Dual Process (16GB+ RAM)**  
```bash
python mac_optimized_embedder.py --mode dual --batch-size 64
```

**Option C: Apple Silicon Optimized (M1/M2/M3 only)**
```bash
python mac_optimized_embedder.py --mode mps --batch-size 128
```

## 📊 Performance Estimates

### **16GB Mac (M2 MacBook Pro)**
- **Strategy:** Single process, batch size 32
- **Time:** ~8-10 hours  
- **Memory usage:** ~12GB peak

### **32GB Mac (M2 Max MacBook Pro)**
- **Strategy:** Dual process, batch size 64
- **Time:** ~4-6 hours
- **Memory usage:** ~20GB peak  

### **64GB Mac (M2 Ultra Mac Studio)**
- **Strategy:** MPS optimized, batch size 256
- **Time:** ~2-3 hours
- **Memory usage:** ~32GB peak

## 🚨 Troubleshooting

### **Out of Memory Errors:**
```bash
# Reduce batch size
python mac_optimized_embedder.py --batch-size 16

# Or use ultra-conservative mode
python mac_optimized_embedder.py --mode conservative --batch-size 8
```

### **Thermal Throttling (Intel Macs):**
```bash
# Monitor CPU temperature
sudo powermetrics -n 1 | grep -i temp

# Use fan control apps: Macs Fan Control, TG Pro
# Process in cooler environment if possible
```

### **Progress Monitoring:**
```bash
# Watch progress logs
tail -f embedding_progress.log

# Check memory usage
top -pid $(pgrep -f mac_optimized_embedder)
```

## 🎯 Final Output

After completion, you'll have:
- **66,716 embeddings** (1024-dimensional BGE-M3 vectors)
- **FAISS HNSW index** for fast similarity search
- **Complete hybrid retrieval system** (dense + BM25)
- **Ready for semantic search** across 70K+ contextual chunks

## 💡 Pro Tips

1. **Run overnight** - embedding generation is I/O bound, perfect for long runs
2. **Use external SSD** if internal storage is limited (200GB+ recommended)  
3. **Close browser tabs** - they consume significant RAM
4. **Enable "Do Not Disturb"** to prevent interruptions
5. **Use Activity Monitor** to watch memory pressure

---

**Next Steps:** Run the optimized Mac embedder and enjoy sub-second search across the entire LessWrong corpus! 🚀