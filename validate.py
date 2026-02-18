import os

# RUTAS A TUS CARPETAS
carpeta_png = r"C:\Users\carlos.delapena\OneDrive - Martinrea Inc\Escritorio\Training\datasets\Best_Seg_DefSide021326\images\train"
carpeta_txt = r"C:\Users\carlos.delapena\OneDrive - Martinrea Inc\Escritorio\Training\datasets\Best_Seg_DefSide021326\labels\train"

# Obtener nombres SIN extensión
png_files = {
    os.path.splitext(f)[0]
    for f in os.listdir(carpeta_png)
    if f.lower().endswith(".png")
}

txt_files = {
    os.path.splitext(f)[0]
    for f in os.listdir(carpeta_txt)
    if f.lower().endswith(".txt")
}

# Comparaciones
con_pareja = png_files & txt_files
png_sin_txt = png_files - txt_files
txt_sin_png = txt_files - png_files

# RESULTADOS
print("📂 RESUMEN DE CARPETAS\n")

print(f"Total archivos PNG: {len(png_files)}")
print(f"Total archivos TXT: {len(txt_files)}")

print("\n🔗 Archivos CON pareja:")
for f in sorted(con_pareja):
    print(f"{f}.png  <->  {f}.txt")

print("\n❌ PNG sin TXT:")
for f in sorted(png_sin_txt):
    print(f"{f}.png")

print("\n❌ TXT sin PNG:")
for f in sorted(txt_sin_png):
    print(f"{f}.txt")

print("\n📊 RESUMEN FINAL")
print(f"Con pareja: {len(con_pareja)}")
print(f"PNG sin pareja: {len(png_sin_txt)}")
print(f"TXT sin pareja: {len(txt_sin_png)}")
