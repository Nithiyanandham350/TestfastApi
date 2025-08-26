# filename: main.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Test FastAPI App", version="1.0")

# --- Example data model ---
class Item(BaseModel):
    name: str
    price: float
    in_stock: bool = True

# --- Root endpoint ---
@app.get("/")
def root():
    return {"message": "🚀 FastAPI test server is running!"}

# --- GET test endpoint ---
@app.get("/ping")
def ping():
    return {"status": "ok", "message": "pong"}

# --- POST test endpoint ---
@app.post("/items/")
def create_item(item: Item):
    return {"message": "Item received", "data": item.dict()}
