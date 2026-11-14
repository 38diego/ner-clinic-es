"""Cálculo del índice de comorbilidad de Charlson de una nota.

Es nuevo frente a la tesis, que asignaba categorías pero no calculaba el índice.

Pesos estándar de Charlson et al. (1987), sin ajuste por edad:

    Charlson ME, Pompei P, Ales KL, MacKenzie CR. A new method of classifying
    prognostic comorbidity in longitudinal studies: development and validation.
    J Chronic Dis. 1987;40(5):373-383. doi:10.1016/0021-9681(87)90171-8

Reglas del cálculo:

- Solo suman las entidades confirmadas; las negadas, las posibles y los antecedentes
  familiares no. `calcular(..., con_posibles=True)` suma también las posibles: es el
  índice si se confirman las sospechas (`Resultado.indice_charlson_si_se_confirman`).
- Cada categoría cuenta una sola vez, aunque la mencionen varias entidades.
- Jerarquía: si en la nota están la forma leve y la severa de diabetes, enfermedad
  hepática o tumor, cuenta solo la de más peso. Como `mapear` asigna una sola categoría
  por texto, la jerarquía aplica entre entidades distintas de la misma nota.

Limitación conocida (#24, pendiente de decisión): la categoría `VIH_SIDA` pesa 6 aunque
la nota solo diga VIH sin SIDA. Adaptaciones posteriores del índice dan al VIH sin SIDA
un peso de 3 y reservan el 6 para el SIDA. Separar la categoría cambia los pesos, así que
por ahora no se hace.
"""

from collections.abc import Iterable

from ner_clinic.tipos import CategoriaCharlson, Entidad, Estado

__all__ = ["JERARQUIAS", "PESOS", "calcular", "calcular_desde_categorias"]

C = CategoriaCharlson

PESOS: dict[CategoriaCharlson, int] = {
    C.INFARTO_MIOCARDIO: 1,
    C.INSUFICIENCIA_CARDIACA: 1,
    C.VASCULAR_PERIFERICA: 1,
    C.CEREBROVASCULAR: 1,
    C.DEMENCIA: 1,
    C.PULMONAR_CRONICA: 1,
    C.REUMATOLOGICA: 1,
    C.ULCERA_PEPTICA: 1,
    C.HEPATICA_LEVE: 1,
    C.DIABETES: 1,
    C.DIABETES_COMPLICADA: 2,
    C.HEMIPLEJIA: 2,
    C.RENAL_CRONICA: 2,
    C.TUMOR_MALIGNO: 2,
    C.HEPATICA_SEVERA: 3,
    C.TUMOR_METASTASICO: 6,
    C.VIH_SIDA: 6,
}
"""Peso de cada categoría según Charlson et al. (1987)."""

JERARQUIAS: dict[CategoriaCharlson, CategoriaCharlson] = {
    C.DIABETES: C.DIABETES_COMPLICADA,
    C.HEPATICA_LEVE: C.HEPATICA_SEVERA,
    C.TUMOR_MALIGNO: C.TUMOR_METASTASICO,
}
"""Forma leve -> forma severa. Si están ambas, la leve no suma."""


def calcular_desde_categorias(categorias: Iterable[CategoriaCharlson]) -> int:
    """Índice de Charlson de un conjunto de categorías (con repeticiones o sin ellas)."""
    presentes = set(categorias)
    presentes -= {leve for leve, severa in JERARQUIAS.items() if severa in presentes}
    return sum(PESOS[c] for c in presentes)


def calcular(entidades: Iterable[Entidad], con_posibles: bool = False) -> int:
    """Índice de Charlson de una nota a partir de sus entidades.

    Suma las confirmadas y, con `con_posibles=True`, también las posibles, con la misma
    jerarquía: una diabetes confirmada y una complicada posible dan 2. Ignora las demás
    entidades y las que no tienen categoría de Charlson.
    """
    estados = {Estado.CONFIRMADA, Estado.POSIBLE} if con_posibles else {Estado.CONFIRMADA}
    return calcular_desde_categorias(
        e.categoria_charlson
        for e in entidades
        if e.estado in estados and e.categoria_charlson is not None
    )
