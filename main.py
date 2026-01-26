#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from dotenv import load_dotenv

from app.config import RAGConfig
from app.service.chatbot import build_app

def main():
    load_dotenv("app/.env")

    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="Pregunta del usuario")
    ap.add_argument("--json", action="store_true", help="Salida JSON")
    args = ap.parse_args()

    cfg = RAGConfig.from_env()
    app = build_app(cfg)
    out = app.handle(args.q)

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(out["answer"])

if __name__ == "__main__":
    main()
