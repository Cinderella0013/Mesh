# ผลยืนยันหลักฐานก่อนส่งอาจารย์ 24 ตุลาคม 2569

รอบนี้ยืนยันการทำงานเดิมและเพิ่มความชัดเจนในการอธิบาย Q กับ gate ไม่ได้เปลี่ยนวิธีหรือเพิ่มคะแนน SAT ไม่มี native attempts, build หรือ training ชุดคะแนนหลักยังเป็นหกวิธี 360 เซลล์ที่เลือกนำเสนอหลังเห็นผล Normal แยกจาก primary ห้า stress conditions

| สิ่งที่ตรวจใหม่ | ผลที่ยืนยันได้ | ขอบเขต |
|---|---|---|
| Planner tests | ผ่าน 18/18 | ซอร์ส RAES เดิม |
| Retained replay | ตรง 120/120 รัน, 1,560 epochs, 23,400 node choices | Hybrid และ Q-neutral อย่างละ 60 รัน; recorded Q ไม่ใช่ inference ใหม่ |
| Q แบบคง actual reservation prefix | เปลี่ยน final choice 138/9,750 หรือ 1.415% | primary stress; gate state เดิม |
| Q แบบวางแผนทั้ง frozen epoch ใหม่ | เปลี่ยน 220/9,750 หรือ 2.256% | ยอมให้ reservation ภายใน epoch เปลี่ยน; ไม่ส่งผลไป observation รอบถัดไป |
| Normal แยก | local 27/1,950; frozen epoch 23/1,950 | ไม่นำไปรวม primary |
| Gate hold ทั้ง 60 รัน | 1,301 ครั้ง: previous REJECTED 1,251 และ STALE 50 | primary 1,194; Normal 107 |
| Permanent policy request ของ hold | ไม่พบ request เก่าค้าง ณ held observation | เชื่อม identity, source และทุก recorded path |

## ความหมายต่อผลงาน

SAT หลักเดิม Hybrid 61.77261520155314% เทียบ raw 35.28323913500535% และ Q-neutral 60.93998305176248% ส่วนเพิ่ม Hybrid–raw เท่ากับ 26.489376066547784 จุดเปอร์เซ็นต์ ส่วนเพิ่มเหนือ Q-neutral เท่ากับ 0.8326321497906548 จุดเปอร์เซ็นต์ ชนะ 29 และแพ้ 21 จาก 50 paired stress cells ผลนี้รองรับการพัฒนาระบบ Hybrid แต่ยังไม่ยืนยันว่า learned GAT เป็นสาเหตุหลัก

เมื่อช่องปัจจุบันยัง allowed และไม่มี gate override ค่า Q budget และ switching penalty เท่ากับ 0.02 พร้อม hysteresis 0.01 การสลับจึงต้องลด pressure penalty มากกว่าอย่างน้อย 0.01 Q ยังเปลี่ยนการเลือกระหว่างช่องทางเลือกได้ การนับ frozen decisions ไม่ได้ทำนายคะแนน SAT ของ trajectory ใหม่ ไม่ใช่หลักฐานว่า Q ไม่มีประโยชน์ทุกกรณี

REJECTED ที่มาก่อน hold ทั้ง 1,251 ครั้งเป็น pause_requested → rejected จบที่ 200 ms ก่อนเริ่ม permanent policy switch และไม่มี switching_started/changed ของ request เดียวกันตลอดรัน ส่วน STALE 50 ครั้งไม่เข้า actuator แล้วตรวจแยก interval ของคำสั่งเก่าทั้งหมดว่าไม่มีค้าง ณ held observation ทุก hold อยู่หลัง previous ack_complete 189.4889–473.5912 ms คำสั่งรอบ hold เป็น unchanged/OK 1,255 ครั้ง และ STALE 46 ครั้ง ไม่มี permanent policy switch ของ held request

จึงอธิบายได้ว่า gate พักหลังความไม่ตรงกันของ requested กับ observed channel แม้ permanent request เดิมจบแล้ว ไม่ควรเรียกทุก hold ว่าคำสั่งเก่ายังค้าง แต่ยังไม่สรุปว่า gate ผิดหรือการเอาออกเพิ่ม QoS เพราะการพักอาจเป็น backoff ต่อ congestion และยังไม่มี native ผลหลังแก้ gate

Scan/calibration retunes เป็นอีก namespace ไม่อยู่ในข้อสรุปว่า permanent request จบแล้ว ไม่มี timeout/ignored ACK ในข้อมูลนี้ Coordinator timeout ไม่ได้ยกเลิก actuator ด้วยตัวมันเอง จึงยังไม่พิสูจน์ alternate timeout, actual late/lost ACK หรือ overlap paths ทุกกรณี

คงผลลบ timing diagnostic เดิม 45.313418520367875% → 40.12774703595256% (−5.185671484415316 จุดเปอร์เซ็นต์) แยกจากค่าเฉลี่ยหลัก ไม่ใช่ balanced timing-controlled comparison ทั้งชุด

## การส่งและสิ่งที่ยังไม่ปิด

ผู้ใช้แจ้งว่าอัปโหลด Drive ครบ 10 ไฟล์แล้ว เราไม่อัปโหลดซ้ำ แต่ยังตรวจ file inventory และ Viewer permissions ผ่านเบราว์เซอร์ไม่ได้ จึงบันทึกเป็น user-reported completion ไม่ใช่ independently verified access

แพ็กเกจ portable นี้ทำซ้ำได้ด้วย Python และ NumPy บน observations ที่แนบ ไม่ได้แทน clean ns-3 build, training/model-selection lineage, external replication หรือ fresh seed/topology confirmation สอง known seeds ไม่รองรับ statistical superiority ยังไม่มี corrected-gate หรือ rescaled-Q service experiment และไม่ได้คืนคู่เทียบที่ผู้ใช้สั่งพักมาจัดอันดับใน paper

Paper สำเนาใหม่คงชื่อเดิม TH Sarabun New สีดำ ไม่มีภาคผนวก ตารางตัวเลขทั้งหกและสมการเดิมไม่เปลี่ยน เพิ่มเฉพาะผล offline และข้อจำกัดที่เกี่ยวข้อง เป็นฉบับให้อาจารย์ตรวจ ยังไม่ใช่ venue clearance
