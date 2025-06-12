# Inhouse Prompt Engineering and RAG

## ข้อกำหนดเบื้องต้น

- Python 3 (แนะนำให้ใช้เวอร์ชัน 3.9 ขึ้นไป และไม่ใช่เวอร์ชัน 3.9.7)
- Poetry (แนะนำให้ใช้เวอร์ชัน 2.1.3)
- Google API Key จาก Google AI Studio

### การติดตั้ง Python

เพื่อตรวจสอบว่าติดตั้ง Python สำเร็จ สามารถใช้คำสั่ง `python --version` ใน Terminal, Command Prompt, หรือ PowerShell
หากไม่พบคำสั่ง `python` สามารถติดตั้งตามขั้นตอนด้านล่าง

1. ดาวน์โหลดตัวติดตั้ง Python เวอร์ชันล่าสุดได้จากเว็บไซต์

- สำหรับ Windows: https://www.python.org/downloads/windows/
- สำหรับ macOS: https://www.python.org/downloads/macos/

2. เรียกใช้ไฟล์ **.exe** หรือ **.pkg** ที่ดาวน์โหลดมา
3. ในหน้าต่างการติดตั้ง ให้ เลือกตัวเลือก **"Add Python to PATH"** ก่อนติดตั้ง **ตัวเลือกนี้มีความสำคัญอย่างยิ่งในการทำให้ระบบรู้จักคำสั่ง Python ในภายหลัง**
4. คลิก **Install Now** เพื่อติดตั้ง Python

### การติดตั้ง Poetry

หากยังไม่ได้ติดตั้ง Poetry สามารถติดตั้งได้จาก [Poetry Website](https://python-poetry.org/docs/)

1. ดาวน์โหลดตัวติดตั้ง Poetry เวอร์ชันล่าสุดได้จากเว็บไซต์: https://python-poetry.org/docs/
2. ดำเนินการติดตั้งตามระบบปฏิบัติการของคุณ

เพื่อตรวจสอบว่าติดตั้ง Poetry สำเร็จ สามารถใช้คำสั่ง `poetry --version` ใน Terminal, Command Prompt, หรือ PowerShell

## วิธีการติดตั้ง

1. Clone repository นี้

```bash
git clone <repository-url>
cd inhouse-prompt-eng-rag
```

2. สร้าง Virtual Environment

```bash
python -m venv .venv
```

3. เปิดใช้งาน Virtual Environment

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

#PowerShell
.venv\Scripts\Activate.ps1
```

4. ติดตั้ง dependencies

### ติดตั้งด้วย pip

```bash
pip install -r requirements.txt
```

### ติดตั้ง dependencies ด้วย Poetry

```bash
poetry install
```

5. สร้างไฟล์ .env ที่ใส่ `GOOGLE_API_KEY`

```bash
GOOGLE_API_KEY=your_google_api_key_here
```

6. สามารถเลือก Run ได้ 2 แบบ

### App UI

```bash
# เมื่อติดตั้ง dependencies ด้วย pip
streamlit run app.py

# เมื่อติดตั้ง dependencies ด้วย poetry
poetry run streamlit run app.py
```

### API

```bash
# เมื่อติดตั้ง dependencies ด้วย pip
python api.py

# เมื่อติดตั้ง dependencies ด้วย poetry
poetry run python api.py
```
