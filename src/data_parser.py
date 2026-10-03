# src/data_parser.py

class AorusDataParser:
    """
    AORUS MASTER 16 AM6H 官方產品規格資料解析器

    資料來源：
    GIGABYTE AORUS MASTER 16 AM6H 官方產品規格
    包含 BZH、BYH、BXH 三種 GPU 配置。
    """

    def __init__(self):
        # ---------------------------------------------------------
        # 基本產品資訊
        # ---------------------------------------------------------
        self.product_info = [
            "產品名稱：GIGABYTE AORUS MASTER 16 AM6H",
            "產品系列：AORUS MASTER 16",
            "產品型號：AORUS MASTER 16 BZH、AORUS MASTER 16 BYH、AORUS MASTER 16 BXH",
        ]

        # ---------------------------------------------------------
        # 作業系統
        # ---------------------------------------------------------
        self.operating_system = [
            "作業系統：Windows 11 Pro",
            "作業系統：Windows 11 Home",
            "作業系統：UEFI Shell OS",
            "GIGABYTE 官方建議商務使用 Windows 11 Pro",
        ]

        # ---------------------------------------------------------
        # CPU
        # ---------------------------------------------------------
        self.cpu = [
            "中央處理器 (CPU)：Intel Core Ultra 9 Processor 275HX",
            "CPU 快取：36MB",
            "CPU 最高時脈：最高 5.4 GHz",
            "CPU 核心數：24 cores",
            "CPU 執行緒數：24 threads",
        ]

        # ---------------------------------------------------------
        # GPU
        # ---------------------------------------------------------
        self.gpu = [
            "AORUS MASTER 16 BZH 顯示晶片：NVIDIA GeForce RTX 5090 Laptop GPU",
            "BZH GPU 記憶體：24GB GDDR7",
            "BZH 最大顯示卡效能：175W Maximum Graphics Power with Dynamic Boost",
            "BZH AI Boost：1797 MHz（1597 MHz Boost Clock + 200 MHz OC）",

            "AORUS MASTER 16 BYH 顯示晶片：NVIDIA GeForce RTX 5080 Laptop GPU",
            "BYH GPU 記憶體：16GB GDDR7",
            "BYH 最大顯示卡效能：175W Maximum Graphics Power with Dynamic Boost",
            "BYH AI Boost：1902 MHz（1702 MHz Boost Clock + 200 MHz OC）",

            "AORUS MASTER 16 BXH 顯示晶片：NVIDIA GeForce RTX 5070 Ti Laptop GPU",
            "BXH GPU 記憶體：12GB GDDR7",
            "BXH 最大顯示卡效能：140W Maximum Graphics Power with Dynamic Boost",
            "BXH AI Boost：1962 MHz（1762 MHz Boost Clock + 200 MHz OC）",

            "GPU 的 AI Boost、時脈與效能可能依使用情境而有所不同。",
        ]

        # ---------------------------------------------------------
        # 顯示器
        # ---------------------------------------------------------
        self.display = [
            "顯示器尺寸：16 吋",
            "顯示器比例：16:10",
            "面板：OLED",
            "解析度：WQXGA 2560×1600",
            "更新率：240Hz",
            "反應時間：1ms",
            "色域：DCI-P3 100%",
            "峰值亮度：500 nits",
            "對比度：1,000,000:1",

            "顯示技術：NVIDIA G-SYNC",
            "顯示技術：NVIDIA Advanced Optimus",
            "顯示技術：VESA DisplayHDR True Black 500",
            "顯示技術：VESA ClearMR 10000",
            "顯示技術：Pantone Validated",
            "顯示技術：TÜV Rheinland Low Blue Light",
            "顯示技術：Dolby Vision",
        ]

        # ---------------------------------------------------------
        # 記憶體
        # ---------------------------------------------------------
        self.memory = [
            "記憶體：最高 64GB DDR5 5600MHz",
            "記憶體插槽：2 x SO-DIMM sockets",
            "記憶體支援擴充：支援透過 2 個 SO-DIMM 插槽進行擴充",
        ]

        # ---------------------------------------------------------
        # 儲存裝置
        # ---------------------------------------------------------
        self.storage = [
            "儲存裝置：最高 4TB PCIe NVMe M.2 SSD",
            "M.2 插槽：1 x PCIe Gen5 M.2 slot",
            "M.2 插槽：1 x PCIe Gen4x4 M.2 slot",
            "儲存空間最高支援：Up to 4TB PCIe NVMe M.2 SSD",
            "實際儲存容量可能依國家及地區而有所不同。",
        ]

        # ---------------------------------------------------------
        # 鍵盤
        # ---------------------------------------------------------
        self.keyboard = [
            "鍵盤：3-zone RGB Backlit Keyboard",
            "鍵程：最高 1.7mm",
            "鍵盤功能：支援 N-Key",
        ]

        # ---------------------------------------------------------
        # 連接埠
        # ---------------------------------------------------------
        self.ports = [
        # 左側
        "左側連接埠：1 x DC in",
        "左側連接埠：1 x RJ-45",
        "左側連接埠：1 x HDMI 2.1",
        "左側連接埠：1 x Type-A support USB3.2 Gen2",
        "左側連接埠：1 x Type-C with Thunderbolt 5 (support USB4, DisplayPort 2.1 and Power Delivery 3.0)",
        
        # 右側
        "右側連接埠：1 x Type-A support USB3.2 Gen2",
        "右側連接埠：1 x Type-C with Thunderbolt 4 (support USB4, DisplayPort 1.4 and Power Delivery 3.0)",
        "右側連接埠：1 x MicroSD (UHS-II)",
        "右側連接埠：1 x Audio Jack support mic/headphone combo",
        ]

        # ---------------------------------------------------------
        # 音效
        # ---------------------------------------------------------
        self.audio = [
            "揚聲器：4 x 2W speakers",
            "麥克風：內建 Microphone",
            "音效技術：Dolby Atmos",
            "音效技術：Smart Amp Technology",
        ]

        # ---------------------------------------------------------
        # 通訊
        # ---------------------------------------------------------
        self.communication = [
            "無線網路：Wi-Fi 7",
            "Wi-Fi 規格：802.11be 2x2",
            "有線網路：1G LAN",
            "Bluetooth：Bluetooth v5.4",
        ]

        # ---------------------------------------------------------
        # 視訊鏡頭
        # ---------------------------------------------------------
        self.camera = [
            "視訊鏡頭：FHD 1080p IR Webcam",
            "麥克風：Built-in array Microphone",
            "Windows Hello：支援 Windows Hello Face Authentication",
        ]

        # ---------------------------------------------------------
        # 安全功能
        # ---------------------------------------------------------
        self.security = [
            "安全功能：Firmware-based TPM",
            "安全功能：支援 Intel Platform Trust Technology (Intel PTT)",
            "生物辨識：支援 Windows Hello Face Authentication",
        ]

        # ---------------------------------------------------------
        # 電池與電源
        # ---------------------------------------------------------
        self.power = [
            "電池：Li-ion 99Wh",
            "電源變壓器：330W AC Adapter",
        ]

        # ---------------------------------------------------------
        # 外觀尺寸與重量
        # ---------------------------------------------------------
        self.physical = [
            "尺寸：357 x 254 x 23~29.9 mm",
            "重量：約 2.5 kg",
            "顏色：Dark Tide",
            "筆記型電腦尺寸可能依配置、製造流程與測量方式而有所不同。",
            "筆記型電腦重量可能依配置、製造流程與測量方式而有所不同。",
        ]

        # ---------------------------------------------------------
        # 官方規格注意事項
        # ---------------------------------------------------------
        self.notes = [
            "產品規格會依各國家地區出貨而有所變動。",
            "實際販售規格應以各地實際出貨狀況為準。",
            "建議向當地經銷商或零售商確認最新產品販售規格。",
            "產品規格、圖片及其他資訊僅供參考。",
            "產品實際規格若與資料不同，應以實際產品為準。",
            "GIGABYTE 保留在任何時間修改產品規格的權利。",
            "產品標示的效能表現可能為晶片廠商或介面組織提出的最大理論值，實際效能可能因規格及設備而有所不同。",
        ]

    def get_chunks(self) -> list[str]:
        """
        將官方規格拆成適合 RAG / Vector Search 的 chunks。

        每一個 chunk 對應一個明確的產品規格概念，
        避免將所有規格塞進單一長文本。
        """

        chunks = []

        # 基本產品資訊
        chunks.extend(self.product_info)

        # 各規格分類
        chunks.extend(self.operating_system)
        chunks.extend(self.cpu)
        chunks.extend(self.gpu)
        chunks.extend(self.display)
        chunks.extend(self.memory)
        chunks.extend(self.storage)
        chunks.extend(self.keyboard)
        chunks.extend(self.ports)
        chunks.extend(self.audio)
        chunks.extend(self.communication)
        chunks.extend(self.camera)
        chunks.extend(self.security)
        chunks.extend(self.power)
        chunks.extend(self.physical)
        chunks.extend(self.notes)

        # ---------------------------------------------------------
        # 綜合產品概述
        # ---------------------------------------------------------
        chunks.append(
            "GIGABYTE AORUS MASTER 16 AM6H 是 AORUS MASTER 系列高效能筆記型電腦，"
            "搭載 Intel Core Ultra 9 275HX 處理器，並提供 NVIDIA GeForce RTX 5090、"
            "RTX 5080 與 RTX 5070 Ti Laptop GPU 三種配置。"
        )

        chunks.append(
            "AORUS MASTER 16 AM6H 配備 16 吋 16:10 OLED WQXGA 2560×1600 "
            "240Hz 顯示器，具備 1ms 反應時間、DCI-P3 100% 色域、500 nits 峰值亮度，"
            "並支援 NVIDIA G-SYNC、Advanced Optimus、DisplayHDR True Black 500、"
            "ClearMR 10000、Pantone Validated、TÜV Rheinland Low Blue Light 與 Dolby Vision。"
        )

        chunks.append(
            "AORUS MASTER 16 AM6H 支援最高 64GB DDR5 5600MHz 記憶體，"
            "具備 2 個 SO-DIMM 插槽，並提供 PCIe Gen5 M.2 與 PCIe Gen4x4 M.2 "
            "插槽，最高支援 4TB PCIe NVMe M.2 SSD。"
        )

        chunks.append(
            "AORUS MASTER 16 AM6H 提供 Thunderbolt 5 與 Thunderbolt 4，"
            "支援 USB4、DisplayPort 與 Power Delivery，並具備 HDMI 2.1、"
            "RJ-45、USB Type-A、MicroSD UHS-II 與複合式音源插孔。"
        )

        chunks.append(
            "AORUS MASTER 16 AM6H 支援 Wi-Fi 7 802.11be 2x2、Bluetooth 5.4 "
            "與 1G LAN，並配備 FHD 1080p IR Webcam、Windows Hello Face Authentication、"
            "Firmware-based TPM 與 Intel Platform Trust Technology。"
        )

        chunks.append(
            "AORUS MASTER 16 AM6H 配備 99Wh Li-ion 電池與 330W AC Adapter，"
            "機身尺寸約 357 x 254 x 23~29.9 mm，重量約 2.5 kg，顏色為 Dark Tide。"
        )

        return chunks