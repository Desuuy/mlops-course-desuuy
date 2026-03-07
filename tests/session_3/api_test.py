import pytest
from fastapi.testclient import TestClient

from scripts.session_3.api import app

# 200 là các mã an toàn
# 400 là các mã lỗi
# 404 là các mã lỗi không tìm thấy
# 405 là các mã lỗi phương thức không hợp lệ
# 406 là các mã lỗi không chấp nhận
# 407 là các mã lỗi không xác thực
# 408 là các mã lỗi thời gian chờ hết hạn
# 409 là các mã lỗi xung đột
# 410 là các mã lỗi đã xóa

@pytest.fixture
def client():
    return TestClient(app)


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, FastAPI!"}


def test_calculate_add(client):
    response = client.post("/calculate", json={"method": "add", "num1": 1, "num2": 2})
    assert response.status_code == 200
    assert response.json() == {"result": 3}


def test_calculate_subtract(client):
    response = client.post(
        "/calculate", json={"method": "subtract", "num1": 1, "num2": 2}
    )
    assert response.status_code == 200
    assert response.json() == {"result": -1}


def test_calculate_multiply(client):
    response = client.post(
        "/calculate", json={"method": "multiply", "num1": 1, "num2": 2}
    )
    assert response.status_code == 200
    assert response.json() == {"result": 2}


def test_calculate_divide(client):
    response = client.post(
        "/calculate", json={"method": "divide", "num1": 1, "num2": 2}
    )
    assert response.status_code == 200
    assert response.json() == {"result": 0.5}


def test_calculate_invalid_method(client):
    response = client.post(
        "/calculate", json={"method": "invalid", "num1": 1, "num2": 2}
    )
    assert response.status_code == 422
