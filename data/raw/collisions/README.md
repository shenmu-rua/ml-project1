# 清洗前的英国道路伤人事故数据（2024—2025）

两份 CSV 是英国交通部 DfT 官方下载文件的逐字节副本，按年份分开；没有删行、补值、改编码，也没有与天气合并。

| 文件 | 行数（不含表头） | 列数 | 字节数 |
|---|---:|---:|---:|
| `dft-road-casualty-statistics-collision-2024.csv` | 100,927 | 44 | 19,208,488 |
| `dft-road-casualty-statistics-collision-2025.csv` | 101,525 | 44 | 19,349,153 |

合计 **202,452 条事故记录**。每行代表一次已记录的伤人事故，不是车辆数或伤亡人数。数据本身不含没有发生事故的道路时段。

`official_data_guide_2025.xlsx` 是官方字段编码说明，`download_manifest.json` 保存下载日期、来源地址、字段、记录数和文件校验信息。事故编号和区域代码建议按文本读取，避免表格软件改写编码；限速单位是 mph。

事故严重程度编码：1=致死、2=重伤、3=轻伤。预测严重程度时需另行审查目标泄漏字段，例如 `enhanced_severity_collision`、`collision_injury_based`、`collision_adjusted_severity_serious` 和 `collision_adjusted_severity_slight`；本目录按用户要求保留原始全部列，尚未构造训练集。

数据提供方：英国交通部 Department for Transport。官方入口：[Road safety open data](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data)。直接下载地址及 SHA-256 见清单。
