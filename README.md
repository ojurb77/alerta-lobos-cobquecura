# Sistema piloto de alerta de varamientos — Cobquecura

Este repositorio es un **andamiaje técnico inicial** para automatizar el cálculo y mostrarlo en un panel web. No es todavía un sistema operativo validado.

## Qué hace
- Consulta diariamente el pronóstico de oleaje de Copernicus Marine.
- Extrae `VTPK` (período de pico, segundos) para una pequeña área frente a Cobquecura.
- Calcula promedios diarios con controles básicos de cobertura.
- Durante la ventana histórica estudiada (16 de enero–15 de febrero), calcula probabilidades con el modelo logístico provisional:
  `p = 1 / (1 + exp(-z))`
  `z = -6.94312994 + 0.10647496 * season_day + 0.43959415 * Tp_previous_day`
- Publica resultados en `docs/data/status.json` para mostrarlos en GitHub Pages.

## Limitaciones críticas
1. El modelo se entrenó con una serie histórica derivada de reanálisis, no con ciclos de pronóstico guardados con anticipación.
2. `VTPK` es el período de pico. Antes de usar alertas operativas hay que confirmar que la agregación espacial/temporal del pronóstico es comparable con la serie histórica usada en el entrenamiento.
3. El umbral de 45% o 50% es experimental. No representa una alerta oficial ni reemplaza la revisión de especialistas.
4. El workflow puede ejecutarse automáticamente, pero la ejecución programada de GitHub Actions no garantiza una hora exacta.
5. La web pública no debe contener credenciales ni información personal.

## Preparación
### 1. Crear el repositorio
Crea un repositorio llamado, por ejemplo, `alerta-lobos-cobquecura`. Si quieres usar GitHub Pages gratis con una cuenta personal gratuita, normalmente debes publicar el sitio desde un repositorio público. El código y el panel serán visibles públicamente; las credenciales deben guardarse solo como secretos del repositorio.

### 2. Subir estos archivos
Descomprime el ZIP y sube el contenido del directorio al repositorio, conservando las carpetas y sus rutas.

### 3. Añadir secretos de Copernicus
En GitHub: **Settings → Secrets and variables → Actions → New repository secret**.
Crea:
- `COPERNICUSMARINE_SERVICE_USERNAME`
- `COPERNICUSMARINE_SERVICE_PASSWORD`

Usa las credenciales de tu propia cuenta de Copernicus Marine. **No las escribas en ningún archivo ni las compartas por chat.**

### 4. Activar GitHub Pages
En **Settings → Pages**, selecciona publicar desde **GitHub Actions**. Luego entra en **Actions** y ejecuta manualmente el workflow `Actualizar panel de Cobquecura` con `Run workflow`.

### 5. Verificar el primer resultado
Abre el log de la ejecución en Actions. El panel debe mostrar si los datos se obtuvieron, si hay cobertura suficiente y si existen predicciones dentro de la ventana estacional. Fuera de esa ventana, no debe presentar una alerta activa.

## Área de descarga
La consulta inicial utiliza aproximadamente:
- longitud: -73.05 a -72.75
- latitud: -36.30 a -35.95
- variable: `VTPK`
- frecuencia: cada 3 horas

Esta zona es una selección inicial de prueba, no una calibración definitiva.

## Qué hacer si falla
- Si aparece un error de autenticación, revisa los secretos; no pegues las credenciales en un issue ni en el código.
- Si faltan datos, no se genera una probabilidad para esa fecha.
- Si Copernicus cambia el catálogo o la disponibilidad, el log de Actions indicará el fallo.
- No interpretes un panel sin datos como “riesgo bajo”: significa que no se pudo calcular.

## Referencias oficiales
- Catálogo y acceso al producto de oleaje: https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_WAV_001_027/description
- Documentación de Copernicus Marine Toolbox: https://help.marine.copernicus.eu/en/articles/7949409-copernicus-marine-toolbox-introduction
- GitHub Pages: https://docs.github.com/en/pages
