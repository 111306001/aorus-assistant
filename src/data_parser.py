class AorusDataParser:
    def __init__(self):
        # 模擬 AORUS MASTER 16 AM6H 的官方規格資料庫（結構化 Key-Value）
        self.specs = {
            "Product Name": "GIGABYTE AORUS MASTER 16 AM6H",
            "Processor": "Intel Core Ultra 7 / Ultra 9 Processor",
            "Graphics": "NVIDIA GeForce RTX Laptop GPU (4GB VRAM target environment test note, actual GPU up to RTX 4080/4090)",
            "Display": "16-inch QHD+ (2560 x 1600) 240Hz IPS-level Panel",
            "Memory": "Up to 64GB DDR5 (Dual Channel)",
            "Storage": "2x M.2 NVMe PCIe 4.0 SSD slots",
            "Keyboard": "AORUS Fusion RGB Per-Key Backlit Keyboard",
            "Battery": "99Whrs",
            "OS": "Windows 11 Pro"
        }

    def get_chunks(self) -> list[str]:
        """將 Key-Value 規格轉化為語意化的文字 Chunk 供檢索"""
        chunks = []
        for key, value in self.specs.items():
            chunks.append(f"The {key} of GIGABYTE AORUS MASTER 16 AM6H is {value}.")
        
        # 加上一段綜合概述
        chunks.append(
            "GIGABYTE AORUS MASTER 16 AM6H is a high-performance gaming laptop featuring "
            "advanced cooling, high refresh rate display, and powerful Intel Core Ultra processors."
        )
        return chunks