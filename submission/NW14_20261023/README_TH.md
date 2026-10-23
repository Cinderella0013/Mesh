# ชุดส่งอาจารย์ NW14 — 23 ตุลาคม 2569

ชุดนี้รวบรวมโค้ด โมเดล ผล และหลักฐานเดิมที่ตรวจสอบได้ ไม่มีการจำลองหรือฝึกใหม่ในการจัดส่งครั้งนี้ ชื่องานในเอกสารคงเดิม

## เปิดอ่านตามลำดับ

1. `documents/` เล่มรายงานฉบับปัจจุบัน, paper และคำตอบอาจารย์ เอกสารถูกคัดลอกโดยไม่แก้เนื้อหา paper/TXT ลงวันที่ 20 ต.ค. ส่วนเล่มเป็นฉบับที่ผู้ใช้แก้ล่าสุด ดู SHA-256 ใน `DOCUMENT_VERSIONS.json`
2. `evidence/tables/METHOD_SUMMARY.csv` ค่าเฉลี่ยหลัก 5 สภาวะ แยกจาก Normal
3. `evidence/tables/CONDITION_LOAD_180.csv` ค่าเฉลี่ยสอง seed ทุกสภาวะและโหลด
4. `evidence/tables/RESULTS_360.csv` ผลราย seed ครบ 360 เซลล์ รวม worst-flow, zero-service, delay, terminal/censored packets และต้นทุนตัดสินใจ
5. `evidence/tables/RAW_PER_FLOW_5400.csv` และตาราง AP/latency/queue รายละเอียด 5,400 flow
6. `project/` ซอร์สและน้ำหนักโมเดลตามเวอร์ชันที่ผูกกับผลจริง แยกแต่ละวิธี ไม่รวมทับไฟล์ต่างเวอร์ชัน
7. `evidence/raw/` หลักฐาน packet, AP และ queue ของทั้ง 360 เซลล์ อยู่ใน ZIP สามไฟล์ ไม่ต้องแตกทั้งหมดเพื่อใช้ตัวตรวจคะแนน

## ขอบเขตและข้อค้นพบ

หกวิธี: GAT+RAES raw, GAT+RAES Hybrid, Q-neutral ablation, GA-Net adaptation, DGN adaptation และ BFS-CA-inspired บน ns-3 AP–STA แบบ single-hop วิธีดัดแปลงไม่ใช่การทำซ้ำบทความครบถ้วนหรือ mesh หลาย hop จริง

ครบ 6 สภาวะ × 5 โหลด (8,12,16,20,24 Mbps ต่อ STA) × 2 known paired seeds (9795002/9795003) × 6 วิธี = 360 เซลล์ ไม่มีการคัดทิ้งเซลล์คะแนนต่ำในหกวิธีนี้

| วิธี | SAT หลัก 5 สภาวะ (%) | Goodput เฉลี่ย (Mbps) |
|---|---:|---:|
| GAT+RAES raw |35.283|58.836|
| GAT+RAES Hybrid |61.773|99.532|
| Q-neutral |60.940|97.702|
| GA-Net adaptation |31.346|49.755|
| DGN adaptation |27.761|42.641|
| BFS-CA-inspired |56.651|94.413|

Hybrid มีค่าเฉลี่ย SAT สูงสุดในหกวิธีที่นำเสนอ แต่ไม่ใช่การชนะทุกกรณี และช่องว่างจาก Q-neutral เพียง 0.833 จุดเปอร์เซ็นต์ยังไม่พิสูจน์ว่าส่วนเรียนรู้จำเป็น สอง seed ที่เคยใช้พัฒนาไม่พออ้างนัยสำคัญหรือการยืนยันกับสภาวะใหม่

ขอบเขตนำเสนอหกวิธีนี้ถูกเลือกหลังเห็นผลชุดเดิม 11 วิธี จึงมีข้อจำกัดด้านการเลือกนำเสนอ ไม่ได้ยืนยันว่า Hybrid ชนะทุกวิธีที่เคยประเมิน

SAT เป็นค่าเฉลี่ยอัตราส่วน delivered bytes ต่อ nominal demand ของ 15 flow เท่ากัน; Goodput เป็น delivered bytes รวมในหน้าต่าง 9 วินาที ตัวเลขทั้งสองไม่ใช่ตัวชี้วัดเดียวกัน p95 เป็น delivered-only ต้องอ่านร่วมกับจำนวนแพ็กเก็ต terminal/censored และ zero-service ห้ามตีความ delay ต่ำว่าให้บริการทุก flow ดี

## ตรวจหลักฐานโดยไม่รัน ns-3

ใช้ Python 3.9+ และ standard library ไม่ต้องติดตั้ง ML frameworks:

```sh
cd evidence
python verify_selection.py --output TABLE_CHECK.json
python verify_selection.py --packets raw/packets_360.zip --ap raw/ap_360.zip --output SAMPLE_CHECK.json
# ตรวจ packet/AP ทุกเซลล์ ใช้เวลามากกว่าตัวอย่าง
python verify_selection.py --packets raw/packets_360.zip --ap raw/ap_360.zip --all --output ALL_RAW_CHECK.json
```

คำสั่งแรกตรวจ grid และคำนวณ SAT/Mbps จากตาราง flow ทั้ง 360 เซลล์ คำสั่งที่สองอ่าน packet จริงของ Hybrid Combined24 seed9795002 (ตัวอย่างที่มี loss/tail สูง ไม่ได้เลือกเฉพาะกรณีดี) ตรวจ hash, UID, เวลา, SAT, Mbps, p95 และ AP join ด้วย `portable_verify.audit` เดิม คำสั่งที่สามตรวจครบ 360 เซลล์ ไม่ใช่การรัน simulator ใหม่

ตรวจไฟล์ที่ดาวน์โหลดกับ `FILE_MANIFEST_SHA256.json` SHA-256 ยืนยันความสอดคล้องของไบต์ที่จัดส่ง ไม่ได้เป็นลายเซ็นรับรองประวัติจากบุคคลภายนอก

## ข้อจำกัดการทำซ้ำโค้ด

ซอร์สและ capsule คงไบต์เดิมเพื่อรักษาการผูกกับหลักฐาน บางโมดูลใช้ร่วมกับวิธีอื่นและจึงมีสาขาโค้ดนอกตารางหลัก ไม่ใช่ผลทดสอบเพิ่มเติมในชุดนี้ ไฟล์ receipt/config ยังเก็บพาธเครื่องต้นทางเพื่อการตรวจสอบ

นี่เป็น source snapshot ที่ตรวจสอบย้อนกลับได้ **ยังไม่ใช่ clean one-command ns-3 build**: ต้องมี ns-3.47 tree และไลบรารีตาม build receipt รวมทั้งแก้การผูกพาธอย่างชัดเจน Selected training เป็น segment สุดท้ายที่ resume มา ไม่ใช่ประวัติ training ครบทุกวิธีหรือทุก segment ดู `project/README.md` ก่อนรัน

ผล native ขึ้นกับเวลาควบคุม/ส่งคำสั่งที่วัดจริงด้วย การมี weights และ seed เดิมอย่างเดียวจึงไม่รับรอง SAT ตรงทุกหลัก การจัดส่งนี้ไม่ได้อ้างว่าทำ clean rebuild หรือยืนยัน generalization ใหม่

โฟลเดอร์เก่าที่ผู้ใช้ห้ามเข้าถึงไม่ได้ถูกเปิดหรือแก้ไขในการจัดส่งนี้
