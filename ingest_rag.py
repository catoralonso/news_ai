"""
ingest_rag.py
─────────────
Populates all ChromaDB collections with seed data for the expo.
Run once from the project root before launching the API:

    python ingest_rag.py

Collections created:
    - news_research       (José + Camila + Manuel)
    - article_generation  (Manuel: writing style)
    - article_published   (Manuel: published articles — shared)
    - fact_checking       (Camila: fake news patterns + trusted sources)
    - reader_interaction  (Mauro: FAQs + recurring questions)
    - social_media        (Asti: high-performing post examples)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core.vector_store import VectorStore
from core.chunker import chunk_document

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/embeddings")


def _upsert(store: VectorStore, docs: list[dict], text_key: str = "content") -> int:
    for doc in docs:
        chunks = chunk_document(doc, text_key=text_key)
        texts = [c["text"] for c in chunks]
        metas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
        store.upsert(texts=texts, metadatas=metas)
    return store.count()


# ─────────────────────────────────────────────────────────────────────────────
# 1. news_research — José
# Topics and angles already covered to avoid repetition
# ─────────────────────────────────────────────────────────────────────────────

NEWS_RESEARCH_DOCS = [
    {
        "title": "Déficit de vitamina D en España",
        "date": "2025-01-10",
        "category": "nutrición",
        "content": (
            "Más del 40% de la población española presenta déficit de vitamina D según un estudio "
            "del Instituto Nacional de Salud. El problema se agrava en invierno por la reducción "
            "de horas de sol. Los expertos recomiendan suplementación entre octubre y marzo para "
            "grupos de riesgo: mayores de 65, personas con piel oscura y quienes trabajan en "
            "interiores. Fuentes consultadas: INS, Sociedad Española de Endocrinología."
        ),
    },
    {
        "title": "Tendencias alimentarias 2025: proteína vegetal",
        "date": "2025-02-01",
        "category": "tendencias",
        "content": (
            "El consumo de proteína vegetal creció un 35% en España durante 2024. Las legumbres, "
            "el tofu y el tempeh lideran el mercado. Los supermercados han triplicado su oferta "
            "de productos plant-based. Nutricionistas advierten de la importancia de combinar "
            "fuentes proteicas vegetales para obtener todos los aminoácidos esenciales. "
            "Garbanzos, lentejas y alubias siguen siendo las opciones más económicas y nutritivas."
        ),
    },
    {
        "title": "Microbiota intestinal y salud mental",
        "date": "2025-02-15",
        "category": "bienestar",
        "content": (
            "Investigadores del CSIC publican estudio sobre el eje intestino-cerebro. La diversidad "
            "microbiana se asocia con menores niveles de ansiedad y depresión. Alimentos fermentados "
            "como yogur, kéfir y chucrut favorecen la salud de la microbiota. La fibra prebiótica "
            "de frutas, verduras y legumbres alimenta las bacterias beneficiosas. "
            "Tema ya cubierto — evitar repetición de enfoque general."
        ),
    },
    {
        "title": "Dieta mediterránea y enfermedades cardiovasculares",
        "date": "2025-03-01",
        "category": "ciencia y evidencia",
        "content": (
            "Metaanálisis en The Lancet: dieta mediterránea reduce 25% riesgo cardiovascular. "
            "Aceite de oliva virgen extra, legumbres y pescado azul son los elementos más protectores. "
            "Abandono progresivo de estos hábitos entre jóvenes españoles preocupa a expertos. "
            "Ángulo ya cubierto desde perspectiva científica — próximo artículo puede enfocarse "
            "en recetas concretas o en el coste económico de la dieta mediterránea."
        ),
    },
    {
        "title": "Azúcar oculto en alimentos procesados",
        "date": "2025-03-10",
        "category": "nutrición",
        "content": (
            "Un adulto español consume de media 95 gramos de azúcar al día, casi el doble del "
            "límite recomendado por la OMS. El azúcar oculto en salsas, panes de molde, yogures "
            "azucarados y cereales de desayuno es el principal responsable. Aprender a leer "
            "etiquetas es clave: sacarosa, jarabe de maíz, dextrosa y maltosa son todos azúcares. "
            "Tema pendiente de desarrollo: impacto en salud dental y metabolismo."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. article_generation — Manuel
# Writing style examples for the newspaper
# ─────────────────────────────────────────────────────────────────────────────

ARTICLE_GENERATION_DOCS = [
    {
        "title": "Guía de estilo Savia",
        "category": "estilo",
        "content": (
            "Savia es un periódico de nutrición cercano, accesible y basado en evidencia. "
            "Tono amigable y directo, evitando tecnicismos innecesarios. "
            "El primer párrafo responde siempre: qué, dónde y a quién afecta. "
            "Títulos concretos orientados al beneficio del lector. Sin titulares alarmistas. "
            "Solo fuentes científicas contrastadas o especialistas identificados. "
            "Entre 3 y 5 párrafos por artículo. Español neutro. "
            "Listas cuando hay más de 3 elementos. "
            "Siempre incluir un consejo práctico al final. "
            "Categorías: nutrición, recetas, bienestar, suplementos, dietas, ciencia y evidencia."
        ),
    },
    {
        "title": "Ejemplo: Los 5 mejores alimentos de otoño",
        "category": "nutrición",
        "content": (
            "Con la llegada del otoño, los mercados locales se llenan de productos frescos que "
            "no solo son más económicos sino también más nutritivos. La calabaza encabeza la lista "
            "por su alto contenido en betacarotenos y vitamina C. Le siguen las setas, fuente "
            "excepcional de vitamina D y proteínas vegetales. Las granadas, ricas en antioxidantes, "
            "los caquis con su poder antiinflamatorio, y las castañas como fuente de energía de "
            "liberación lenta completan el quinteto. Todos disponibles por menos de 3€ el kilo. "
            "Consejo práctico: prepara una crema de calabaza con jengibre para aprovechar sus "
            "propiedades antiinflamatorias en los días más fríos."
        ),
    },
    {
        "title": "Ejemplo: Qué comer antes de hacer ejercicio",
        "category": "bienestar",
        "content": (
            "Comer correctamente antes del ejercicio puede marcar la diferencia entre una sesión "
            "de entrenamiento eficaz y un bajón de energía a mitad del camino. La dietista Laura "
            "Gómez del Centro de Nutrición Deportiva de Madrid recomienda consumir carbohidratos "
            "de liberación lenta entre 1 y 2 horas antes: avena, arroz integral o una tostada "
            "de pan de centeno con mantequilla de cacahuete. Evitar grasas y fibra en exceso "
            "justo antes del ejercicio para no entorpecer la digestión. "
            "Consejo práctico: un plátano con un puñado de almendras es la opción perfecta "
            "si solo tienes 30 minutos antes de entrenar."
        ),
    },
    {
        "title": "Ejemplo: Todo lo que debes saber sobre el hierro",
        "category": "suplementos",
        "content": (
            "La anemia por déficit de hierro afecta a 1 de cada 4 mujeres en edad fértil en España. "
            "El hierro hemo, presente en carnes rojas, pescado y marisco, se absorbe hasta tres "
            "veces mejor que el hierro no hemo de las legumbres y verduras de hoja verde. "
            "Para mejorar la absorción del hierro vegetal, combínalo siempre con vitamina C: "
            "zumo de limón en las lentejas o pimiento rojo en la ensalada de espinacas. "
            "Evita tomar café o té durante la comida si consumes fuentes de hierro, ya que "
            "los taninos reducen su absorción hasta un 60%. "
            "Consejo práctico: consulta con tu médico antes de suplementarte — el exceso "
            "de hierro puede ser perjudicial."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# 3. article_published — Manuel (shared with José, Mauro, Asti)
# Already published articles
# ─────────────────────────────────────────────────────────────────────────────

ARTICLE_PUBLISHED_DOCS = [
    {
        "title": "Vitamina D: el nutriente del que probablemente carezcas este invierno",
        "category": "nutrición",
        "angle": "Guía práctica para prevenir el déficit en los meses fríos",
        "local_relevance_score": 0.9,
        "content": (
            "Con la llegada del invierno y la reducción de horas de sol, muchos españoles "
            "reducen su síntesis natural de vitamina D sin saberlo. Esta vitamina esencial "
            "para la salud ósea, el sistema inmune y el estado de ánimo se produce principalmente "
            "a través de la exposición solar en la piel. "
            "La doctora Marta Sánchez, nutricionista del Hospital La Paz, explica que entre "
            "octubre y marzo la radiación solar en España no es suficiente para que la piel "
            "sintetice vitamina D de forma efectiva, especialmente por encima del paralelo 40. "
            "Los alimentos más ricos en vitamina D son el salmón, las sardinas en conserva, "
            "el atún, los huevos y los lácteos enriquecidos. Sin embargo, la dieta por sí "
            "sola raramente cubre las necesidades diarias recomendadas de 600-800 UI. "
            "Consejo práctico: aprovecha los días soleados para salir al menos 20 minutos "
            "entre las 11h y las 15h con brazos y cara descubiertos. Consulta a tu médico "
            "si sospechas un déficit — un análisis de sangre puede confirmarlo."
        ),
    },
    {
        "title": "La microbiota intestinal: cómo cuidar a los billones de aliados en tu intestino",
        "category": "bienestar",
        "angle": "Ciencia actualizada y consejos prácticos para mejorar tu flora intestinal",
        "local_relevance_score": 0.85,
        "content": (
            "Tu intestino alberga más de 100 billones de microorganismos que pesan colectivamente "
            "casi 2 kilos. Esta comunidad, conocida como microbiota intestinal, influye en tu "
            "digestión, tu sistema inmune y, según investigaciones recientes, incluso en tu estado "
            "de ánimo a través del eje intestino-cerebro. "
            "Un estudio publicado en Nature Medicine por investigadores del CSIC revela que las "
            "personas con mayor diversidad microbiana presentan menores niveles de marcadores "
            "inflamatorios y mejor salud metabólica. La clave para una microbiota saludable "
            "está en la variedad: cuantos más tipos distintos de plantas y fermentados consumas, "
            "más diversa y resiliente será tu comunidad microbiana. "
            "Los probióticos naturales del yogur con cultivos activos, el kéfir, el chucrut y "
            "el kimchi introducen bacterias beneficiosas. Los prebióticos del ajo, la cebolla, "
            "el puerro, los espárragos y la avena las alimentan. "
            "Consejo práctico: proponte añadir una nueva verdura a tu dieta cada semana. "
            "La diversidad vegetal es el mejor predictor de una microbiota saludable."
        ),
    },
    {
        "title": "Proteína vegetal: la guía definitiva para no necesitar carne",
        "category": "dietas",
        "angle": "Cómo obtener proteína completa solo con alimentos vegetales",
        "local_relevance_score": 0.88,
        "content": (
            "La proteína vegetal está de moda, pero muchas personas no saben cómo combinar "
            "fuentes para obtener todos los aminoácidos esenciales que el cuerpo no puede "
            "fabricar por sí mismo. La buena noticia: no necesitas combinarlos en la misma "
            "comida, sino a lo largo del día. "
            "Las legumbres son ricas en lisina pero bajas en metionina. Los cereales integrales "
            "tienen el perfil inverso. Por eso la combinación tradicional de lentejas con arroz "
            "o hummus con pan de pita es nutricionalmente brillante. "
            "El tofu, el tempeh y la soja son proteínas completas por sí solas — contienen "
            "todos los aminoácidos esenciales en proporciones adecuadas. "
            "Para alcanzar los 0,8 gramos de proteína por kilo de peso recomendados, "
            "una persona de 70 kg necesita unos 56 gramos diarios: equivalente a una lata "
            "de garbanzos más un vaso de leche de soja más una ración de tofu. "
            "Consejo práctico: empieza incluyendo un día de proteína vegetal a la semana "
            "y aumenta gradualmente según te vayas familiarizando con las alternativas."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# 4. fact_checking — Camila
# Known fake news patterns and trusted sources in nutrition
# ─────────────────────────────────────────────────────────────────────────────

FACT_CHECKING_DOCS = [
    {
        "title": "Bulo: el limón en ayunas cura el cáncer",
        "category": "fake_news",
        "content": (
            "BULO VERIFICADO. Afirmación falsa: beber agua con limón en ayunas cura o previene "
            "el cáncer. Origen: cadenas de WhatsApp y blogs sin fuente científica. "
            "Realidad: el limón es un alimento saludable rico en vitamina C pero no tiene "
            "propiedades anticancerígenas demostradas. Ningún alimento por sí solo cura el cáncer. "
            "El pH del limón se neutraliza completamente en el estómago y no afecta el pH sanguíneo. "
            "Fuentes de verificación: AECC (Asociación Española Contra el Cáncer), "
            "Cochrane Database, PubMed. Nivel de peligrosidad: ALTO — puede llevar a retrasar "
            "tratamientos médicos reales."
        ),
    },
    {
        "title": "Bulo: los lácteos causan cáncer",
        "category": "fake_news",
        "content": (
            "BULO PARCIALMENTE FALSO. Afirmación exagerada: los lácteos causan cáncer. "
            "Realidad matizada: existe asociación estadística entre consumo muy elevado de "
            "lácteos y ligero aumento de riesgo de cáncer de próstata en algunos estudios. "
            "Sin embargo, la evidencia es inconsistente y el efecto es pequeño. "
            "Los lácteos no causan cáncer en el consumo habitual recomendado (2-3 raciones/día). "
            "La OMS los clasifica como alimentos seguros. "
            "Fuentes confiables: OMS, EFSA, Harvard T.H. Chan School of Public Health. "
            "Nivel de peligrosidad: MEDIO — puede generar déficit de calcio si se eliminan sin alternativa."
        ),
    },
    {
        "title": "Bulo: el gluten es tóxico para todos",
        "category": "fake_news",
        "content": (
            "BULO FALSO. Afirmación incorrecta: el gluten es dañino para todas las personas. "
            "Realidad: el gluten solo es perjudicial para personas con enfermedad celíaca "
            "(1% de la población), sensibilidad al gluten no celíaca (estimado 6%) y alergia al trigo. "
            "Para el resto de la población, el gluten es perfectamente seguro. "
            "Los productos sin gluten no son más saludables para personas sin estas condiciones "
            "y suelen tener más azúcar y grasas para compensar la textura. "
            "Fuentes confiables: FACE (Federación de Asociaciones de Celíacos de España), "
            "Sociedad Española de Gastroenterología. "
            "Nivel de peligrosidad: BAJO-MEDIO — genera gasto innecesario y puede reducir "
            "ingesta de fibra si se eliminan cereales integrales."
        ),
    },
    {
        "title": "Fuentes confiables en nutrición",
        "category": "trusted_sources",
        "content": (
            "FUENTES VERIFICADAS Y CONFIABLES EN NUTRICIÓN: "
            "Nivel 1 (máxima confianza): OMS/WHO, EFSA (Agencia Europea de Seguridad Alimentaria), "
            "AESAN (Agencia Española de Seguridad Alimentaria), PubMed/NCBI, Cochrane Reviews. "
            "Nivel 2 (alta confianza): Harvard T.H. Chan School of Public Health, "
            "Sociedad Española de Dietética y Ciencias de la Alimentación (SEDCA), "
            "Fundación Española de la Nutrición (FEN), CSIC. "
            "Nivel 3 (confianza media — verificar): colegios de dietistas-nutricionistas, "
            "hospitales universitarios, revistas científicas indexadas (IF > 2). "
            "FUENTES NO CONFIABLES: blogs sin autor identificado, influencers sin titulación, "
            "estudios financiados por la industria sin revisión independiente, "
            "cadenas de WhatsApp, titulares sensacionalistas sin enlace al estudio original."
        ),
    },
    {
        "title": "Bulo: el ayuno intermitente es peligroso",
        "category": "fake_news",
        "content": (
            "BULO FALSO. Afirmación exagerada: el ayuno intermitente daña el metabolismo y "
            "provoca trastornos alimentarios. "
            "Realidad: el ayuno intermitente es seguro para adultos sanos según múltiples estudios. "
            "El protocolo 16:8 (16 horas de ayuno, 8 de alimentación) es el más estudiado "
            "y muestra beneficios en control de peso, sensibilidad a la insulina y marcadores "
            "inflamatorios en personas sin patologías previas. "
            "No está recomendado para: embarazadas, personas con diabetes tipo 1, "
            "menores de 18 años, personas con historial de trastornos alimentarios. "
            "Fuentes: New England Journal of Medicine, revisión 2022 sobre ayuno intermitente. "
            "Nivel de peligrosidad: BAJO — puede ser contraproducente en poblaciones específicas."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# 5. reader_interaction — Mauro
# FAQs and recurring reader questions
# ─────────────────────────────────────────────────────────────────────────────

READER_INTERACTION_DOCS = [
    {
        "category": "suplementos",
        "content": (
            "Q: ¿Debo tomar suplementos de vitamina D en invierno?\n"
            "A: Depende de tu nivel en sangre. Lo ideal es hacerte una analítica para conocer "
            "tu nivel de 25-OH vitamina D. Si está por debajo de 20 ng/mL tienes déficit. "
            "En España, entre octubre y marzo la mayoría de personas en latitudes del centro "
            "y norte no sintetizan suficiente vitamina D solar. Grupos de riesgo (mayores de 65, "
            "personas con piel oscura, quienes trabajan en interiores) suelen beneficiarse de "
            "suplementación de 1000-2000 UI/día. Consulta siempre a tu médico antes de suplementarte."
        ),
    },
    {
        "category": "nutrición",
        "content": (
            "Q: ¿Cuánta proteína necesito al día?\n"
            "A: La recomendación general es 0,8 gramos por kilo de peso corporal al día para "
            "adultos sedentarios. Si haces ejercicio regularmente, entre 1,2 y 1,6 g/kg. "
            "Si entrenas fuerza con intensidad, hasta 2 g/kg. Por ejemplo, una persona de 70 kg "
            "que va al gimnasio 3 veces por semana necesita entre 84 y 112 gramos de proteína diarios. "
            "Puedes obtenerlos de carne, pescado, huevos, lácteos, legumbres o una combinación de todos."
        ),
    },
    {
        "category": "dietas",
        "content": (
            "Q: ¿Es saludable el ayuno intermitente?\n"
            "A: Para adultos sanos, sí. El protocolo más estudiado es el 16:8: comes en una "
            "ventana de 8 horas y ayunas 16 horas (incluyendo las horas de sueño). "
            "Los beneficios más documentados son: ayuda en el control de peso, mejora la "
            "sensibilidad a la insulina y reduce marcadores inflamatorios. "
            "No está recomendado si estás embarazada, tienes diabetes tipo 1, eres menor de edad "
            "o tienes historial de trastornos alimentarios. Consulta con un profesional antes de empezar."
        ),
    },
    {
        "category": "bienestar",
        "content": (
            "Q: ¿Qué alimentos ayudan a dormir mejor?\n"
            "A: Varios alimentos contienen nutrientes que favorecen el sueño. "
            "El triptófano, precursor de la melatonina, está presente en el pavo, el pollo, "
            "los lácteos, los huevos, los plátanos y las nueces. "
            "El magnesio, que relaja el sistema nervioso, abunda en las semillas de calabaza, "
            "los frutos secos, las legumbres y el chocolate negro. "
            "Evita la cafeína después de las 14h y cenas copiosas 2-3 horas antes de dormir. "
            "Un vaso de leche tibia o una infusión de valeriana o manzanilla pueden ayudar."
        ),
    },
    {
        "category": "nutrición",
        "content": (
            "Q: ¿Es mejor el aceite de oliva virgen extra o el normal?\n"
            "A: El aceite de oliva virgen extra (AOVE) es claramente superior. "
            "A diferencia del aceite de oliva refinado, el AOVE se obtiene por extracción en frío "
            "sin procesos químicos, conservando todos sus polifenoles y antioxidantes. "
            "Estos compuestos son los responsables de sus beneficios cardiovasculares demostrados. "
            "El AOVE soporta perfectamente las temperaturas de cocción habituales (hasta 180-190°C). "
            "El precio mayor está justificado por su calidad nutricional — es una inversión en salud."
        ),
    },
    {
        "category": "nutrición",
        "content": (
            "Q: ¿Qué es el índice glucémico y para qué sirve?\n"
            "A: El índice glucémico (IG) mide la velocidad con la que un alimento eleva el azúcar "
            "en sangre comparado con la glucosa pura. Alto IG (>70): pan blanco, arroz blanco, "
            "patatas fritas. Medio IG (56-69): arroz integral, plátano maduro. "
            "Bajo IG (<55): legumbres, avena, manzana, yogur natural. "
            "Los alimentos de bajo IG generan saciedaad más duradera y evitan picos de insulina. "
            "Para personas con diabetes o resistencia a la insulina, controlar el IG es especialmente "
            "importante. Para la población general, priorizar alimentos integrales y legumbres es "
            "suficiente sin necesidad de calcular el IG de cada alimento."
        ),
    },
    {
        "category": "suplementos",
        "content": (
            "Q: ¿El colágeno en polvo funciona realmente?\n"
            "A: La evidencia científica es limitada pero prometedora. Los suplementos de colágeno "
            "hidrolizado se absorben como aminoácidos y péptidos que el cuerpo puede usar para "
            "sintetizar su propio colágeno. Estudios de calidad moderada muestran mejoras en "
            "elasticidad de la piel y reducción de dolor articular tras 8-12 semanas de uso. "
            "Sin embargo, el efecto no está garantizado para todos. "
            "Alternativa más barata: una dieta rica en vitamina C (necesaria para sintetizar colágeno), "
            "proteínas de calidad y zinc produce resultados similares en muchas personas."
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# 6. social_media — Asti
# High-performing post examples per platform
# ─────────────────────────────────────────────────────────────────────────────

SOCIAL_MEDIA_DOCS = [
    {
        "platform": "twitter",
        "engagement": "alto",
        "category": "nutrición",
        "content": (
            "El 70% de tu sistema inmune vive en tu intestino. "
            "Lo que comes hoy, lo sientes mañana. 🦠 #microbiota #nutricion #savia"
        ),
    },
    {
        "platform": "twitter",
        "engagement": "muy_alto",
        "category": "datos",
        "content": (
            "Dato: necesitas 20 minutos al sol para sintetizar vitamina D. "
            "En invierno en España eso no es suficiente. "
            "Habla con tu médico antes de suplementarte. ☀️ #vitaminaD #salud"
        ),
    },
    {
        "platform": "twitter",
        "engagement": "alto",
        "category": "mito",
        "content": (
            "MITO: el gluten es malo para todo el mundo. "
            "REALIDAD: solo es dañino para el 1% con celiaquía y el 6% con sensibilidad. "
            "El resto puede comerlo sin problema. #nutricion #mitos #savia"
        ),
    },
    {
        "platform": "instagram",
        "engagement": "muy_alto",
        "category": "bienestar",
        "content": (
            "¿Sabías que comer mal no siempre significa comer poco saludable? 🥦\n\n"
            "Muchas veces el problema no está en lo que comemos, sino en cómo, cuándo y "
            "con qué combinamos los alimentos. Pequeños cambios en la rutina diaria pueden "
            "tener un impacto enorme en cómo nos sentimos.\n\n"
            "Desde añadir más fibra al desayuno hasta hidratarnos mejor durante el día. "
            "La clave está en la constancia, no en la perfección.\n\n"
            "¿Qué pequeño cambio te comprometes a hacer esta semana? 👇"
        ),
    },
    {
        "platform": "instagram",
        "engagement": "alto",
        "category": "recetas",
        "content": (
            "3 desayunos que te darán energía hasta la hora de comer 🌅\n\n"
            "1️⃣ Avena con plátano y nueces — fibra + proteína + omega-3\n"
            "2️⃣ Tostada de centeno con aguacate y huevo — grasas buenas + proteína completa\n"
            "3️⃣ Yogur griego con frutos rojos y semillas — probióticos + antioxidantes\n\n"
            "Todos listos en menos de 10 minutos. ¿Cuál es tu favorito?"
        ),
    },
    {
        "platform": "newsletter",
        "engagement": "alto",
        "category": "ciencia",
        "content": (
            "Esta semana en Savia exploramos la relación entre la dieta mediterránea y la "
            "prevención de enfermedades cardiovasculares. Un nuevo metaanálisis publicado "
            "en The Lancet confirma lo que los nutricionistas llevan años diciendo: "
            "el aceite de oliva virgen extra, las legumbres y el pescado azul son aliados "
            "insustituibles de tu corazón. Lee el artículo completo en nuestra web."
        ),
    },
    {
        "platform": "newsletter",
        "engagement": "muy_alto",
        "category": "tendencias",
        "content": (
            "Proteína vegetal, microbiota y vitamina D: los tres grandes temas de nutrición "
            "que están marcando 2025. Esta semana te traemos una guía práctica para incorporar "
            "más proteína vegetal en tu día a día sin renunciar al sabor ni a la saciedad. "
            "Además, Mauro responde las preguntas más frecuentes de nuestros lectores sobre "
            "suplementación y ayuno intermitente."
        ),
    },
    {
        "platform": "carousel",
        "engagement": "muy_alto",
        "category": "nutrición",
        "content": (
            "5 señales de que necesitas más magnesio — muchas personas tienen déficit sin saberlo. "
            "Desliza para descubrirlas. "
            "1: Calambres musculares frecuentes. "
            "2: Dificultad para conciliar el sueño. "
            "3: Fatiga constante sin causa aparente. "
            "4: Irritabilidad o ansiedad. "
            "5: Dolor de cabeza recurrente. "
            "Fuentes de magnesio: semillas de calabaza, almendras, espinacas, chocolate negro."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Savia RAG — ingesting seed data\n")

    # 1. news_research
    store = VectorStore("news_research", f"{PERSIST_DIR}/news_research")
    count = _upsert(store, NEWS_RESEARCH_DOCS)
    print(f"  news_research       → {count} chunks")

    # 2. article_generation (Manuel style)
    store = VectorStore("article_generation", f"{PERSIST_DIR}/article_generation")
    count = _upsert(store, ARTICLE_GENERATION_DOCS)
    print(f"  article_generation  → {count} chunks")

    # 3. article_published (shared)
    store = VectorStore("article_published", f"{PERSIST_DIR}/article_published")
    count = _upsert(store, ARTICLE_PUBLISHED_DOCS)
    print(f"  article_published   → {count} chunks")

    # 4. fact_checking (Camila)
    store = VectorStore("fact_checking", f"{PERSIST_DIR}/fact_checking")
    count = _upsert(store, FACT_CHECKING_DOCS)
    print(f"  fact_checking       → {count} chunks")

    # 5. reader_interaction (Mauro)
    store = VectorStore("reader_interaction", f"{PERSIST_DIR}/reader_interaction")
    count = _upsert(store, READER_INTERACTION_DOCS)
    print(f"  reader_interaction  → {count} chunks")

    # 6. social_media (Asti)
    store = VectorStore("social_media", f"{PERSIST_DIR}/social_media")
    count = _upsert(store, SOCIAL_MEDIA_DOCS)
    print(f"  social_media        → {count} chunks")

    print("\nAll collections ready.")
    print(f"ChromaDB persisted at: {PERSIST_DIR}")


if __name__ == "__main__":
    main()
