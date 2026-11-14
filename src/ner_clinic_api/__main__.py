"""`python -m ner_clinic_api`: corre la API con uvicorn y el access log desactivado.

Host y puerto salen de `HOST` (por defecto 127.0.0.1) y `PORT` (por defecto 8000; Cloud
Run pone `PORT`).
"""

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "ner_clinic_api.app:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        # El access log registra la IP de cada visitante: no hace falta y se apaga.
        access_log=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
