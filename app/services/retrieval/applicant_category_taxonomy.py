from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class ApplicantCategoryGroupDefinition:
    code: str
    label: str
    question_patterns: tuple[str, ...]
    evidence_terms: tuple[str, ...]
    weight: float
    examples: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class ApplicantCategoryGroupMatch:
    code: str
    label: str
    matched_patterns: list[str]
    matched_terms: list[str]


def normalize_text(value: str | None) -> str:
    text = str(value or "").replace("\xa0", " ").replace("ё", "е").lower()
    text = re.sub(r"[^0-9a-zа-я\s\-\"№]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_applicant_category_groups() -> list[ApplicantCategoryGroupDefinition]:
    """
    Примерная смысловая карта категорий заявителей.

    Основа — категории из таблиц identifiers текущего корпуса НПА. Это не
    юридическая классификация и не вывод о праве заявителя, а словарь для
    первичного подбора возможных мер по широким вопросам.
    """
    return [
        ApplicantCategoryGroupDefinition(
            code="vov_participant",
            label="участники Великой Отечественной войны",
            question_patterns=(
                r"\bучастник\s+вов\b",
                r"\bучастник\w*\s+велик\w*\s+отечественн\w*\s+войн\w*\b",
                r"\bветеран\w*\s+велик\w*\s+отечественн\w*\s+войн\w*\b",
            ),
            evidence_terms=("участник великой отечественной войны", "участники великой отечественной войны", "ветеран великой отечественной войны", "великой отечественной войны"),
            weight=4.5,
            examples=("участник Великой Отечественной войны",),
        ),
        ApplicantCategoryGroupDefinition(
            code="vov_disabled",
            label="инвалиды Великой Отечественной войны",
            question_patterns=(
                r"\bинвалид\w*\s+вов\b",
                r"\bинвалид\w*\s+велик\w*\s+отечественн\w*\s+войн\w*\b",
            ),
            evidence_terms=("инвалид великой отечественной войны", "инвалиды великой отечественной войны", "инвалидов великой отечественной войны"),
            weight=4.7,
            examples=("инвалид Великой Отечественной войны",),
        ),
        ApplicantCategoryGroupDefinition(
            code="blockade_siege",
            label="жители блокадного Ленинграда и осаждённых городов",
            question_patterns=(
                r"\bблокадн\w*\s+ленинград",
                r"\bжител\w*\s+блокадн\w*\s+ленинград",
                r"\bосажденн\w*\s+(?:севастопол|сталинград)",
            ),
            evidence_terms=("блокадн", "жителю блокадного ленинграда", "житель осажденного севастополя", "житель осажденного сталинграда"),
            weight=4.2,
        ),
        ApplicantCategoryGroupDefinition(
            code="home_front_worker",
            label="труженики тыла",
            question_patterns=(
                r"\bтруженик\w*\s+тыла\b",
                r"\bработал\w*\s+в\s+тылу\b",
                r"\bпроработал\w*\s+в\s+тылу\b",
            ),
            evidence_terms=("труженик", "труженики тыла", "проработавшие в тылу", "работы в тылу"),
            weight=4.0,
        ),
        ApplicantCategoryGroupDefinition(
            code="minor_prisoner",
            label="бывшие несовершеннолетние узники",
            question_patterns=(
                r"\bнесовершеннолетн\w*\s+узник",
                r"\bузник\w*\s+(?:концлагер|фашизм)",
                r"\bконцлагер",
            ),
            evidence_terms=("несовершеннолетний узник", "несовершеннолетние узники", "узник", "концлагер", "гетто", "принудительного содержания"),
            weight=4.0,
        ),
        ApplicantCategoryGroupDefinition(
            code="combat_veteran",
            label="ветераны боевых действий",
            question_patterns=(r"\bветеран\w*\s+боев\w*\s+действ",),
            evidence_terms=("ветеран боевых действий", "ветераны боевых действий", "боевых действий"),
            weight=4.1,
        ),
        ApplicantCategoryGroupDefinition(
            code="military_or_svo",
            label="военнослужащие, участники СВО и связанные с ними лица",
            question_patterns=(
                r"\bвоеннослуж",
                r"\bвоенн\w*\s+служб",
                r"\bсво\b",
                r"\bспециальн\w*\s+военн\w*\s+операц",
                r"\bмобилизованн",
            ),
            evidence_terms=("военнослуж", "военной службы", "сво", "специальной военной операции", "мобилизован"),
            weight=3.9,
        ),
        ApplicantCategoryGroupDefinition(
            code="family_of_fallen",
            label="члены семей погибших или умерших защитников и военнослужащих",
            question_patterns=(
                r"\bсемь\w*\s+погибш",
                r"\bродител\w*\s+погибш",
                r"\bдет\w*\s+погибш",
                r"\bвдов\w*\s+(?:военнослуж|участник|инвалид)",
                r"\bпогиб\w*\s+(?:военнослуж|на\s+сво|участник)",
            ),
            evidence_terms=("члены семей погиб", "семей погиб", "вдовы", "вдовцы", "дети погиб", "родители погиб", "погибших защитников"),
            weight=4.3,
        ),
        ApplicantCategoryGroupDefinition(
            code="honorary_donor",
            label="почётные доноры",
            question_patterns=(
                r"\bдонор\b",
                r"\bпочетн\w*\s+донор",
                r"\bпочетный\s+донор",
            ),
            evidence_terms=("донор", "почетный донор", "почетного донора", "нагрудным знаком почетный донор"),
            weight=4.4,
            examples=("Почетный донор России", "Почетный донор СССР"),
        ),
        ApplicantCategoryGroupDefinition(
            code="rehabilitated",
            label="реабилитированные и пострадавшие от политических репрессий",
            question_patterns=(
                r"\bреабилитир",
                r"\bполитическ\w*\s+репресс",
                r"\bпострадавш\w*\s+от\s+политическ\w*\s+репресс",
            ),
            evidence_terms=("реабилитир", "политических репресс", "пострадавшими от политических репрессий"),
            weight=4.0,
        ),
        ApplicantCategoryGroupDefinition(
            code="labor_veteran",
            label="ветераны труда",
            question_patterns=(r"\bветеран\w*\s+труда\b",),
            evidence_terms=("ветеран труда", "ветераны труда", "приравненные к ветеранам труда"),
            weight=4.0,
        ),
        ApplicantCategoryGroupDefinition(
            code="regional_labor_veteran",
            label="ветераны труда Красноярского края",
            question_patterns=(r"\bветеран\w*\s+труда\s+края\b", r"\bветеран\w*\s+труда\s+красноярск"),
            evidence_terms=("ветеран труда края", "ветераны труда красноярского края"),
            weight=4.4,
        ),
        ApplicantCategoryGroupDefinition(
            code="long_service_awards",
            label="лица с длительным стажем, наградами и почётными званиями",
            question_patterns=(
                r"\bстаж\w*\s+(?:работ|служб)",
                r"\b(?:35|40|25|20)\s+лет\s+стаж",
                r"\bнагражден",
                r"\bпочетн\w*\s+зван",
                r"\bорден",
                r"\bмедал",
            ),
            evidence_terms=("стаж", "продолжительность работы", "поощр", "награжден", "почетн", "орден", "медал", "знаками отличия"),
            weight=3.6,
        ),
        ApplicantCategoryGroupDefinition(
            code="disabled_person",
            label="инвалиды",
            question_patterns=(r"\bинвалид\b", r"\bинвалидност", r"\bограничени\w*\s+здоров"),
            evidence_terms=("инвалид", "инвалиды", "инвалидности", "ограниченн", "опорно двигательного аппарата"),
            weight=3.7,
        ),
        ApplicantCategoryGroupDefinition(
            code="disabled_child",
            label="дети-инвалиды и их родители",
            question_patterns=(r"\bребен\w*\s*-?\s*инвалид", r"\bдет\w*\s*-?\s*инвалид"),
            evidence_terms=("ребенок инвалид", "ребенка инвалида", "дети инвалиды", "детей инвалидов", "родитель ребенка инвалида"),
            weight=4.1,
        ),
        ApplicantCategoryGroupDefinition(
            code="pensioner",
            label="пенсионеры, предпенсионеры и нетрудоспособные граждане",
            question_patterns=(r"\bпенсионер", r"\bпенси[яюи]\b", r"\bпредпенсион", r"\bнетрудоспособ"),
            evidence_terms=("пенсионер", "пенсию", "пенсии", "пенсион", "нетрудоспособ", "достигшие возраста 55", "достигшие возраста 60"),
            weight=3.2,
        ),
        ApplicantCategoryGroupDefinition(
            code="large_family",
            label="многодетные семьи и родители в многодетных семьях",
            question_patterns=(
                r"\bмногодет",
                r"\b(?:трое|троих|тремя|трех|трех|3)\s+(?:несовершеннолетн\w*\s+)?дет",
                r"\bдет\w*\s+(?:трое|троих|тремя|трех|трех|3)\b",
            ),
            evidence_terms=("многодет", "троих и более детей", "трех и более детей", "3 и более детей", "пять и более детей"),
            weight=3.8,
        ),
        ApplicantCategoryGroupDefinition(
            code="single_parent",
            label="единственный родитель / неполная семья",
            question_patterns=(
                r"\bмать\s*-?\s*одиноч",
                r"\bотец\s*-?\s*одиноч",
                r"\bодинок\w*\s+(?:мать|отец|родител)",
                r"\bединственн\w*\s+родител",
                r"\bнеполн\w*\s+сем",
            ),
            evidence_terms=(
                "мать одиночка",
                "отец одиночка",
                "одинокая мать",
                "одинокий отец",
                "одинокий родитель",
                "одинокая родитель",
                "единственный родитель",
                "единственного родителя",
                "неполной семьи",
                "неполная семья",
                "один из родителей неполной семьи",
            ),
            weight=3.5,
        ),
        ApplicantCategoryGroupDefinition(
            code="low_income",
            label="малоимущие граждане и семьи",
            question_patterns=(
                r"\bмалоимущ",
                r"\bнизк\w*\s+доход",
                r"\bдоход\w*\s+ниже",
                r"\bне\s+хватает\s+денег",
                r"\bнужда(?:юсь|емся|ющийся|ющаяся|ющиеся)",
            ),
            evidence_terms=("малоимущ", "доход", "среднедуш", "нуждаем", "прожиточного минимума"),
            weight=3.4,
        ),
        ApplicantCategoryGroupDefinition(
            code="family_with_children",
            label="семьи с детьми, родители и законные представители детей",
            question_patterns=(r"\bдет\w*\b", r"\bребен\w*\b", r"\bродител", r"\bсемь\w*\s+с\s+детьми"),
            evidence_terms=("дет", "ребен", "родител", "семь", "законный представитель", "усынов", "опека", "попечител", "приемн"),
            weight=2.6,
        ),
        ApplicantCategoryGroupDefinition(
            code="orphans",
            label="дети-сироты и дети, оставшиеся без попечения родителей",
            question_patterns=(r"\bдет\w*\s*-?\s*сирот", r"\bбез\s+попечения\s+родител"),
            evidence_terms=("дети сироты", "детей сирот", "без попечения родителей", "лица из числа детей сирот"),
            weight=4.0,
        ),
        ApplicantCategoryGroupDefinition(
            code="student",
            label="обучающиеся и студенты",
            question_patterns=(r"\bстудент", r"\bобучающ", r"\bочная\s+форм", r"\bучусь\b"),
            evidence_terms=("обучающ", "очной форме", "образовательн", "студент"),
            weight=3.0,
        ),
        ApplicantCategoryGroupDefinition(
            code="rural_specialist",
            label="специалисты, работающие и проживающие в сельской местности",
            question_patterns=(
                r"\bсельск\w*\s+местност",
                r"\bработа\w*\s+в\s+селе",
                r"\bпедагог",
                r"\bмедицинск\w*\s+работник",
                r"\bветеринар",
                r"\bработник\w*\s+культур",
            ),
            evidence_terms=("сельск", "поселках городского типа", "педагог", "медицин", "ветеринар", "культуры", "социальные работники"),
            weight=3.2,
        ),
        ApplicantCategoryGroupDefinition(
            code="emergency_victim",
            label="пострадавшие от чрезвычайной ситуации, пожара или утраты имущества",
            question_patterns=(
                r"\bчс\b",
                r"\bчрезвычайн\w*\s+ситуац",
                r"\bпожар",
                r"\bсгорел\w*\s+(?:дом|жилье|жилье|квартир)",
                r"\bутрат\w*\s+имуществ",
            ),
            evidence_terms=("чрезвычайн", "чс", "пострадавш", "утрат", "имущество первой необходимости", "зона чрезвычайной ситуации"),
            weight=4.3,
        ),
        ApplicantCategoryGroupDefinition(
            code="hardship",
            label="граждане в трудной жизненной ситуации / нуждающиеся",
            question_patterns=(
                r"\bтжс\b",
                r"\bтрудн\w*\s+жизненн\w*\s+ситуац",
                r"\bпомогите\b",
                r"\bнечего\s+есть\b",
                r"\bнужна\s+помощь\b",
                r"\bдайте\s+денег\b",
                r"\bне\s+хватает\s+денег\b",
            ),
            evidence_terms=("трудн", "жизненн", "ситуац", "нуждаем", "экстренн", "адресная материальная помощь"),
            weight=3.6,
        ),
        ApplicantCategoryGroupDefinition(
            code="food_need",
            label="нуждаемость в продуктах питания или первой необходимости",
            question_patterns=(
                r"\bнечего\s+есть\b",
                r"\bнет\s+(?:еды|продуктов|денег\s+на\s+еду)\b",
                r"\bпомогите\b.*\b(?:есть|еда|продукт|питани)\b",
                r"\bпомощь\s+на\s+(?:еду|продукт|питани)\b",
            ),
            evidence_terms=("трудн", "жизненн", "ситуац", "нуждаем", "малоимущ", "экстренн", "адресная материальная помощь"),
            weight=3.7,
        ),
        ApplicantCategoryGroupDefinition(
            code="fuel_need",
            label="нуждаемость в твёрдом топливе, дровах, угле или отоплении",
            question_patterns=(
                r"\bдров",
                r"\bугол",
                r"\bугл[яе]м",
                r"\bтоплив",
                r"\bтв[её]рд\w*\s+топлив",
                r"\bотоплен",
            ),
            evidence_terms=("дров", "угол", "топлив", "твердое топливо", "твёрдое топливо", "отоплен"),
            weight=3.4,
        ),
        ApplicantCategoryGroupDefinition(
            code="assistive_device_need",
            label="потребность в техническом средстве реабилитации",
            question_patterns=(
                r"\bтср\b",
                r"\bтехническ\w*\s+средств\w*\s+реабилитац",
                r"\bсредств\w*\s+реабилитац",
                r"\bкресл\w*\s*-?\s*(?:каталк|коляск)",
                r"\bколяск",
                r"\bслухов\w*\s+аппарат",
            ),
            evidence_terms=("тср", "техническое средство реабилитации", "технические средства реабилитации", "средство реабилитации", "кресло каталка", "кресло коляска", "коляска", "слуховой аппарат"),
            weight=4.2,
        ),
        ApplicantCategoryGroupDefinition(
            code="dental_prosthesis_need",
            label="потребность в зубных или стоматологических протезах",
            question_patterns=(
                r"\bзубн\w*\s+протез",
                r"\bстоматологическ\w*\s+протез",
                r"\bзубопротез",
                r"\bпротезирован\w*\s+зуб",
            ),
            evidence_terms=("зубопротез", "зубные протезы", "зубной протез", "стоматологические протезы", "стоматологический протез", "протезирование зубов"),
            weight=4.2,
        ),
        ApplicantCategoryGroupDefinition(
            code="free_travel_need",
            label="потребность в льготном или бесплатном проезде",
            question_patterns=(
                r"\bбесплатн\w*\s+проезд",
                r"\bльготн\w*\s+проезд",
                r"\bпроездн\w*\s+удостовер",
                r"\bсоциальн\w*\s+карт",
                r"\bсоцкарт",
                r"\bпроезд\w*\s+(?:в\s+)?(?:автобус|общественн\w*\s+транспорт)",
                r"\bавтобус",
            ),
            evidence_terms=("бесплатный проезд", "льготный проезд", "проездное удостоверение", "проездные удостоверения", "социальная карта", "соцкарта", "общественный транспорт", "автобус", "проезд"),
            weight=3.9,
        ),
        ApplicantCategoryGroupDefinition(
            code="school_need",
            label="подготовка детей к школе",
            question_patterns=(r"\bсобрать\s+дет\w*\s+в\s+школ", r"\bдет\w*\s+в\s+школ", r"\bшкольн\w*\s+(?:форм|одежд|принадлежност)"),
            evidence_terms=("школ", "школь", "школьная форма", "школьные принадлежности"),
            weight=3.2,
        ),
        ApplicantCategoryGroupDefinition(
            code="health_condition",
            label="граждане с заболеваниями или медицинскими основаниями",
            question_patterns=(r"\bонколог", r"\bдиабет", r"\bзаболеван", r"\bмедицинск\w*\s+показан"),
            evidence_terms=("онколог", "диабет", "заболеван", "медицин", "санаторно курорт"),
            weight=3.0,
        ),
        ApplicantCategoryGroupDefinition(
            code="housing_or_utilities",
            label="граждане по жилищным и коммунальным основаниям",
            question_patterns=(r"\bжиль", r"\bжилищ", r"\bкоммунальн", r"\bсубсид", r"\bкапитальн\w*\s+ремонт"),
            evidence_terms=("жил", "жилищ", "коммун", "субсид", "ремонт жилого помещения"),
            weight=2.8,
        ),
        ApplicantCategoryGroupDefinition(
            code="burial_or_memorial",
            label="лица, взявшие расходы или обязанности по погребению, памятникам, могилам",
            question_patterns=(r"\bпогреб", r"\bпохорон", r"\bмогил", r"\bпамятник", r"\bнадгроб"),
            evidence_terms=("погреб", "могил", "памятник", "надгроб", "благоустройству могил"),
            weight=3.6,
        ),
        ApplicantCategoryGroupDefinition(
            code="hero_family",
            label="Герои труда и члены их семей",
            question_patterns=(r"\bгеро\w*\s+труда", r"\bтрудов\w*\s+слав"),
            evidence_terms=("герой труда", "героя труда", "героя социалистического труда", "трудовой славы"),
            weight=3.7,
        ),
        ApplicantCategoryGroupDefinition(
            code="representative",
            label="законные и уполномоченные представители заявителей",
            question_patterns=(r"\bпредставител", r"\bдоверенн", r"\bзаконн\w*\s+представител"),
            evidence_terms=("представител", "законный представитель", "уполномоченный представитель", "доверенн"),
            weight=2.4,
        ),
        ApplicantCategoryGroupDefinition(
            code="territory_evenkiya_taymyr",
            label="территориальные категории Эвенкии и Таймыра",
            question_patterns=(r"\bэвенк", r"\bтаймыр", r"\bдолгано", r"\bненец"),
            evidence_terms=("эвенк", "таймыр", "долгано", "ненец"),
            weight=2.2,
        ),
    ]


def extract_question_groups(question_text: str | None) -> list[ApplicantCategoryGroupMatch]:
    text = normalize_text(question_text)
    matches: list[ApplicantCategoryGroupMatch] = []
    for definition in build_applicant_category_groups():
        matched_patterns = [
            pattern
            for pattern in definition.question_patterns
            if re.search(pattern, text, flags=re.IGNORECASE)
        ]
        if not matched_patterns:
            continue
        matches.append(
            ApplicantCategoryGroupMatch(
                code=definition.code,
                label=definition.label,
                matched_patterns=matched_patterns,
                matched_terms=list(definition.evidence_terms),
            )
        )
    return deduplicate_group_matches(matches)


def classify_applicant_category_text(category_text: str | None) -> list[ApplicantCategoryGroupMatch]:
    text = normalize_text(category_text)
    matches: list[ApplicantCategoryGroupMatch] = []
    if not text:
        return matches
    for definition in build_applicant_category_groups():
        matched_terms = [term for term in definition.evidence_terms if normalize_text(term) and normalize_text(term) in text]
        if not matched_terms:
            continue
        matches.append(
            ApplicantCategoryGroupMatch(
                code=definition.code,
                label=definition.label,
                matched_patterns=[],
                matched_terms=matched_terms,
            )
        )
    return deduplicate_group_matches(matches)


def deduplicate_group_matches(matches: Iterable[ApplicantCategoryGroupMatch]) -> list[ApplicantCategoryGroupMatch]:
    result: list[ApplicantCategoryGroupMatch] = []
    seen: set[str] = set()
    for match in matches:
        if match.code in seen:
            continue
        seen.add(match.code)
        result.append(match)
    return result


def build_group_index() -> dict[str, ApplicantCategoryGroupDefinition]:
    return {definition.code: definition for definition in build_applicant_category_groups()}


def group_definitions_as_json() -> list[dict[str, Any]]:
    return [
        {
            "code": definition.code,
            "label": definition.label,
            "question_patterns": list(definition.question_patterns),
            "evidence_terms": list(definition.evidence_terms),
            "weight": definition.weight,
            "examples": list(definition.examples),
        }
        for definition in build_applicant_category_groups()
    ]
