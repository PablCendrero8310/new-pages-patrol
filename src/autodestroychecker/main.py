# Copyright (C) 2026 Pablo Cendrero
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the COPYING file for more details.

import asyncio
import logging
import re
from collections import deque
from typing import Dict

import mwparserfromhell
import numpy as np
import pywikibot
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "gravitee-io/detoxify-onnx"

CONTENT_TOXICITY_THRESHOLD = 0.85
TITLE_TOXICITY_THRESHOLD = 0.90

MAX_PROCESSED_CHANGES = 1000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s → %(message)s",
)

# --------------------------------------------------
# Load AI model ONCE
# --------------------------------------------------

logging.info("Loading Detoxify model...")

model = ORTModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    file_name="model.quant.onnx",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

logging.info("Model loaded.")

# --------------------------------------------------
# Pywikibot
# --------------------------------------------------

site = pywikibot.Site("es", "wikipedia")

# Cambios ya procesados
processed_changes = set()

# Mantiene el orden de los últimos cambios procesados
processed_order = deque()


def should_skip_user(user: pywikibot.User):
    try:
        groups = user.groups()

        return "sysop" in groups or "autopatrolled" in groups

    except Exception as e:
        logging.error(
            "Could not get groups for %s: %s",
            user,
            e,
        )

        return False


def mark_processed(rcid):
    processed_changes.add(rcid)
    processed_order.append(rcid)

    # Si superamos el límite, eliminamos el más antiguo
    if len(processed_order) > MAX_PROCESSED_CHANGES:
        old_rcid = processed_order.popleft()
        processed_changes.discard(old_rcid)


# --------------------------------------------------
# Analyze text
# --------------------------------------------------


def analyze(text: str) -> Dict[str, float]:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    outputs = model(**inputs)
    logits = outputs.logits

    probs = 1 / (1 + np.exp(-logits))

    result = {}

    for i, prob in enumerate(probs[0]):
        label = model.config.id2label[i]
        result[label] = float(prob)

    return result


# --------------------------------------------------
# Get page
# --------------------------------------------------


def get_page(title: str):
    page = pywikibot.Page(site, title)

    try:
        wikitext = page.text
    except Exception as e:
        logging.error(
            "Could not retrieve %s: %s",
            title,
            e,
        )
        return None

    # Divide el wikitexto por encabezados
    pattern = r"^(={2,})\s*(.*?)\s*\1\s*$"

    matches = list(
        re.finditer(
            pattern,
            wikitext,
            flags=re.MULTILINE,
        )
    )

    sections = []

    # --------------------------------------------------
    # Introducción
    # --------------------------------------------------

    if matches:
        intro = wikitext[: matches[0].start()].strip()
    else:
        intro = wikitext.strip()

    if intro:
        code = mwparserfromhell.parse(intro)

        text = code.strip_code()
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            sections.append(
                {
                    "title": "Introducción",
                    "text": text,
                }
            )

    # Si no hay secciones, terminamos aquí
    if not matches:
        return sections

    # --------------------------------------------------
    # Secciones
    # --------------------------------------------------

    for i, match in enumerate(matches):
        section_title = match.group(2).strip()

        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(wikitext)

        section_wikitext = wikitext[start:end].strip()

        if not section_wikitext:
            continue

        # Convierte wikitexto a texto plano
        code = mwparserfromhell.parse(section_wikitext)

        text = code.strip_code()

        # Normaliza espacios
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            sections.append(
                {
                    "title": section_title,
                    "text": text,
                }
            )

    return sections


# --------------------------------------------------
# Process new page
# --------------------------------------------------


async def process_change(change):

    # Obtener ID único del cambio
    rcid = change.get("rcid")
    user = pywikibot.User(site, change.get("user"))
    title = change.get("title")
    if rcid is None or not title or not user:
        return

    # Si ya hemos procesado este cambio, ignorarlo
    if rcid in processed_changes:
        return
    # Marcarlo como procesado
    mark_processed(rcid)

    if await asyncio.to_thread(
        should_skip_user,
        user,
    ):
        return
    logging.info(
        "New page → %s",
        title,
    )

    sections = await asyncio.to_thread(
        get_page,
        title,
    )

    if sections is None:
        return

    if not sections:
        logging.info(
            "%s → empty page",
            title,
        )
        return

    logging.info(
        "%s → %d sections",
        title,
        len(sections),
    )

    scores = []

    for section in sections:
        # Analizar título de sección
        if section["title"]:
            result = analyze(section["title"])
            scores.append(result["toxicity"])

        # Analizar texto
        if section["text"]:
            result = analyze(section["text"])
            scores.append(result["toxicity"])

    if not scores:
        content_score = 0.0

    else:
        average = np.average(scores)
        maximum = np.max(scores)

        content_score = round(
            np.average(
                [average, maximum],
                weights=[0.25, 0.75],
            ),
            2,
        )

    logging.info(
        "%s → score: %.2f",
        title,
        content_score,
    )

    # Eliminar el espacio de nombres del título
    plain_title = title.split(":", 1)[-1]

    title_score = analyze(plain_title)["toxicity"]

    # Solo continuar si al menos uno de los dos
    # supera su threshold
    if (
        content_score < CONTENT_TOXICITY_THRESHOLD
        and title_score < TITLE_TOXICITY_THRESHOLD
    ):
        return

    logging.warning(
        "%s → HIGH TOXICITY DETECTED (content: %.2f, title: %.2f)",
        title,
        content_score,
        title_score,
    )

    if not await asyncio.to_thread(
        edit_page,
        title,
    ):
        return
    await asyncio.to_thread(notify_user, user, title)


def notify_user(user: pywikibot.User, title: str):
    talk_page = user.getUserTalkPage()

    message = f"{{subst:Aviso destruir|1={title}|2=g2}} ~~~~"

    talk_page.save(
        summary="Bot: aviso sobre página creada",
        watch=None,
        minor=False,
        appendtext=message,
    )


# --------------------------------------------------
# Edit page
# --------------------------------------------------


def edit_page(title: str):
    page = pywikibot.Page(site, title)

    try:
        text = page.text

        if has_destroy_template(text):
            logging.info(
                "%s → already has destroy template",
                title,
            )
            return False

        new_text = "{{destruir|bot=PCendrerBOT|g2}}\n" + text

        page.text = new_text

        page.save(
            summary="Bot: posible contenido tóxico detectado",
            minor=False,
        )

        logging.info(
            "%s → page edited successfully",
            title,
        )
    except Exception as e:
        logging.error(
            "Could not edit %s: %s",
            title,
            e,
        )
        return False
    return True


# --------------------------------------------------
# Check if destroy template exists
# --------------------------------------------------


def has_destroy_template(texto):
    code = mwparserfromhell.parse(texto)

    return any(
        template.name.matches("Destruir") for template in code.filter_templates()
    )


# --------------------------------------------------
# Check recent changes with Pywikibot
# --------------------------------------------------


async def check_recent_changes():

    logging.info("Starting to check recent changes...")

    while True:
        try:
            # Pywikibot es síncrono,
            # así que se ejecuta en otro thread
            changes = await asyncio.to_thread(
                lambda: list(
                    site.recentchanges(
                        namespaces=[0, 1, 2, 3],
                        changetype="new",
                        total=20,
                    )
                )
            )

            for change in changes:
                await process_change(change)

        except Exception:
            logging.exception("Error checking recent changes")

        # Esperar antes de volver a consultar
        await asyncio.sleep(10)


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    asyncio.run(check_recent_changes())
