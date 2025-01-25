from googletrans import Translator
from asyncio import run
async def translate_text(text, src_lang, dest_lang):
  try:
    translator = Translator()
    translation = await translator.translate(text, src=src_lang, dest=dest_lang)
    return translation.text
  except Exception as e:
    print(f"Translation failed: {e}")
    return None
if __name__ == "__main__":
  source_text = input("Enter the text to be translate here: ")
  source_lang = input("Enter the source language code here: ")
  dest_lang = input("Enter the destination language code here: ")
  translated_text = run(translate_text(source_text, source_lang, dest_lang))
  if translated_text:
    print(f"Translated text: {translated_text}")
  else:
    print("Translation failed.")