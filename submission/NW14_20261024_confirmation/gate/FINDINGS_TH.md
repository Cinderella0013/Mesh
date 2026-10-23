# ผลตรวจ gate แบบ offline

ตรวจ fresh PASS จาก `actuation_hybrid` 60 runs × 13 epochs × 15 nodes = **11,700 node-epochs** โดยอ่านเฉพาะ native ของแถว Hybrid ใน registry และตรวจ hash ของ input ทั้ง 420 ไฟล์ ไม่คำนวณ packet metrics ใหม่

- พบ held cases **1,301**: previous REJECTED **1,251**, previous STALE **50**
- primary 50 runs มี 9,750 node-epochs / held 1,194; normal reference 10 runs มี 1,950 node-epochs / held 107 แยกไว้ใน aggregate
- สถานะทุก request: OK 10,255, REJECTED 1,310, STALE 135 ไม่มี BUSY/ILLEGAL
- REPORT/COMMAND/ACK ทุก request มี enqueue → tx → rx แบบ identity ตรงกันและไม่ซ้ำ รวมอย่างละ 35,100 events มี `report_complete`, `command_submitted`, `ack_complete` อย่างละ 780 และ `ack_partial` 11,700 ไม่พบ timeout, ignored ACK, invalid/enqueue_failed, missing path หรือเวลาย้อนลำดับ

**ข้อสรุปที่หลักฐานรองรับ:** สำหรับ held 1,301 cases ไม่พบ permanent policy request รุ่นเก่าค้างอยู่ ณ observation และไม่พบ switching_started/changed ของ previous request เดียวกันภายหลัง completion หรือภายหลัง hold ภายในทั้ง recorded run

previous REJECTED ทั้ง 1,251 cases มี controller path `pause_requested → rejected` และ endpoint command RX ถึง completion/ACK enqueue เท่ากับ 200,000,000 ns ไม่มี `switching_started` หรือ `changed` ของ request นั้นตลอดไฟล์ จึงเป็น **pre-switch terminal rejection** ตาม source: `Poll` ตั้ง ChannelSettings เฉพาะ quiescence branch แล้ว `return` หลัง schedule FinishSwitch (wifi_threeway.cc:1173–1181); deadline rejection resume, trace, clear `pending.active`, Complete(REJECTED), return (1183–1196) จึงไม่ได้อาศัย management REJECTED เพียงอย่างเดียวเพื่ออนุมานเรื่อง PHY

previous STALE 50 cases ไม่มี controller event ของ request นั้น และ age ณ endpoint อยู่ที่ 235,242,200–509,438,700 ns ซึ่งเกิน deadline 200 ms ใน `Endpoint` (management_link.h:188–191) ก่อนเรียก actuator จากนั้นตรวจแยก interval ของ policy request รุ่นก่อนหน้าทุกตัวแล้วว่าปิดก่อน held observation จึงไม่ได้ตีความ STALE ว่ายกเลิก actuator เก่า

ทุก held case มี previous `ack_complete` ก่อน observation ถัดไป โดยห่าง **189,488,900–473,591,200 ns** และ selected target เท่ากับช่องที่ observation อ่านได้และ previous ACK readback ตรงกัน Current held request 1,255 cases เป็น `unchanged/OK`; อีก 46 เป็น `STALE` โดยไม่เข้า actuator ทั้ง 1,301 ไม่มี permanent policy switching event ของ held request

Source ตรง commit `f5921db0d72edc70e0c6a5163d4f669ed8eb43c5` และ git diff ของ source ที่ตรวจเป็นค่าว่าง:

- wifi_threeway.cc SHA256 `4b31d60c33c8a8b7f59ec5932e5280439ecc4d0246068a627342f12b7df31cd5` (raw CRLF); normalized LF `f2a9931df94d7ef08357c343381d59776057fce2713c886af554f614139bdb9e`
- management_link.h SHA256 `4dc8e3a7f7b4b68955c9cda7531a1a3e77f11128fc84b316416c28c06637fbd6`

ไฟล์หลัก: `node_epoch_requests.csv` เก็บ identity และเวลา exact ต่อ request ทั้ง 11,700; `held_cases.csv` เชื่อม previous/current identity พร้อม timestamps ทั้ง 1,301; `aggregate.json` เก็บ totals และ normal reference แยก; `source_hash_line_map.json` ระบุ source lines; `input_hashes.csv` เก็บ SHA256 ทุก input

ทดสอบรันซ้ำด้วย portable registry `portable_replay/REGISTRY_120.csv` fresh PASS: เลือกเฉพาะ Hybrid 60 runs แล้ว SHA256/จำนวน byte ของ input 420 ไฟล์ตรงกับต้นทางทุกไฟล์ ผล counts และ held-by-condition ตรงกัน เก็บหลักฐานใน `portability_verification.json`, `aggregate_portable.json`, `input_hashes_portable.csv` จากนั้นรัน default ต้นทางซ้ำอีกครั้ง PASS เพื่อให้ตารางหลักชี้กลับหลักฐานต้นทาง

ข้อจำกัด: เป็นผลเฉพาะ 60 recorded runs นี้ ไม่ใช่คำรับรองทุก runtime path หรือ hardware และไม่ใช่ข้อเสนอ fixed gate implementation การ timeout ของ coordinator เพียงอย่างเดียวไม่ cancel actuator (management_link.h:203–205); alternate timeout path ยังไม่ถูกพิสูจน์ เพราะไม่ปรากฏใน log ชุดนี้

Scan/calibration ใช้ namespace แยก; `ScanRetune` (wifi_threeway.cc:1060–1070) ตั้งช่องระหว่าง acquisition และไม่ได้ emit channel-control event ต่อ retune จึง **ไม่อ้างว่าตัววิทยุไม่มีการ retune ใด ๆ หลัง rejected/hold** Trace `timeS` มีหก significant digits จึงจับคู่ด้วย rounding interval; เวลา completion exact ใช้ ACK stamp/enqueue โดยอาศัย synchronous callback ใน source ไม่ขยายผลเป็น PHY telemetry ที่ไม่มีบันทึก

ไม่ได้แก้ native/build/training/policy, ไม่ rerun simulator, ไม่เพิ่มข้ออ้าง SAT improvement และไม่อ่าน packet raw metrics ความเสี่ยงที่ยังเหลือคือ observability ของ alternate/unrecorded paths และ scan retunes ซึ่งอยู่นอกข้อสรุป permanent policy request
