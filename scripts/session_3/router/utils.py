from fastapi import APIRouter

utils_router = APIRouter(prefix="/utils")


@utils_router.get("/health")
def health(dump_input: int):
    if dump_input > 10:
        return {"message": f"This is a large number. Your input was {dump_input}"}
    else:
        return {"message": f"This is a small number. Your input was {dump_input}"}
