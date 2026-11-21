import streamlit as st
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import defaultdict

# =====================================================
# 1. CONFIGURATION DE L'APPLICATION
# =====================================================

st.set_page_config(
    page_title="Simulation d'évacuation",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Simulation stochastique d'évacuation")
st.subheader(
    "Modélisation et simulation de l'évacuation des personnes "
    "dans un ensemble de bâtiments en situation d'urgence"
)

st.markdown("""
Cette application simule l'évacuation de personnes réparties dans
trois bâtiments indépendants. Elle permet d'étudier l'influence
des temps de réaction, des vitesses de déplacement et de la congestion.
""")

# =====================================================
# 2. PARAMÈTRES DU MODÈLE
# =====================================================

N_BUILDINGS = 3
N_FLOORS = 5
ROOMS_PER_FLOOR = 4
STAIRS_PER_BUILDING = 2
EXITS_PER_BUILDING = 2

DT = 0.5

# Distances hypothétiques en mètres
ROOM_TO_CORRIDOR = 8
CORRIDOR_TO_STAIR = 12
STAIR_PER_FLOOR = 3.5
GROUND_TO_EXIT = 8

# Capacités et circulation
ROOM_CAPACITY = 12
CORRIDOR_CAPACITY = 8
STAIR_CAPACITY = 10
EXIT_CAPACITY = 12

# =====================================================
# 3. CONSTRUCTION DES GRAPHES DES BÂTIMENTS
# =====================================================

def build_building_graph(building_id):

    G = nx.Graph()

    # Création des étages, salles et couloirs
    for floor in range(N_FLOORS):

        corridor = f"B{building_id}_F{floor}_C"

        G.add_node(
            corridor,
            kind="corridor",
            floor=floor
        )

        for room in range(ROOMS_PER_FLOOR):

            room_node = f"B{building_id}_F{floor}_R{room}"

            G.add_node(
                room_node,
                kind="room",
                floor=floor
            )

            G.add_edge(
                room_node,
                corridor,
                length=ROOM_TO_CORRIDOR,
                capacity=ROOM_CAPACITY,
                headway=0.5
            )

        # Deux escaliers par étage
        for stair in range(STAIRS_PER_BUILDING):

            stair_node = f"B{building_id}_F{floor}_S{stair}"

            G.add_node(
                stair_node,
                kind="stair",
                floor=floor
            )

            G.add_edge(
                corridor,
                stair_node,
                length=CORRIDOR_TO_STAIR,
                capacity=CORRIDOR_CAPACITY,
                headway=0.7
            )

    # Connexion verticale des escaliers
    for floor in range(N_FLOORS - 1):

        for stair in range(STAIRS_PER_BUILDING):

            lower = f"B{building_id}_F{floor}_S{stair}"
            upper = f"B{building_id}_F{floor + 1}_S{stair}"

            G.add_edge(
                lower,
                upper,
                length=STAIR_PER_FLOOR,
                capacity=STAIR_CAPACITY,
                headway=0.8
            )

    # Deux sorties de secours au rez-de-chaussée
    for exit_number in range(EXITS_PER_BUILDING):

        exit_node = f"B{building_id}_EXIT{exit_number}"

        G.add_node(
            exit_node,
            kind="exit",
            floor=0
        )

        for stair in range(STAIRS_PER_BUILDING):

            stair_node = f"B{building_id}_F0_S{stair}"

            G.add_edge(
                stair_node,
                exit_node,
                length=GROUND_TO_EXIT,
                capacity=EXIT_CAPACITY,
                headway=1.0
            )

    return G


def build_all_graphs():

    return {
        b: build_building_graph(b)
        for b in range(1, N_BUILDINGS + 1)
    }


# =====================================================
# 4. GÉNÉRATION ALÉATOIRE DES PERSONNES
# =====================================================

def generate_people(graphs, number_people, seed):

    random.seed(seed)
    np.random.seed(seed)

    people = []

    for pid in range(number_people):

        building = random.choice(list(graphs.keys()))
        G = graphs[building]

        rooms = [
            node for node, data in G.nodes(data=True)
            if data["kind"] == "room"
        ]

        origin = random.choice(rooms)

        exits = [
            node for node, data in G.nodes(data=True)
            if data["kind"] == "exit"
        ]

        # Choisir une sortie parmi les sorties les plus proches
        distances = {
            exit_node: nx.shortest_path_length(
                G, origin, exit_node, weight="length"
            )
            for exit_node in exits
        }

        minimum = min(distances.values())

        possible_exits = [
            e for e, d in distances.items()
            if d == minimum
        ]

        destination = random.choice(possible_exits)

        path = nx.shortest_path(
            G,
            origin,
            destination,
            weight="length"
        )

        reaction_time = random.uniform(5, 35)

        speed = np.clip(
            np.random.normal(1.20, 0.20),
            0.55,
            2.0
        )

        people.append({
            "id": pid,
            "building": building,
            "origin": origin,
            "path": path,
            "reaction": reaction_time,
            "speed": speed,
            "ready_time": reaction_time,
            "index": 0,
            "moving": False,
            "remaining": 0.0,
            "edge": None,
            "done": False,
            "evacuation_time": None,
            "waiting_time": 0.0
        })

    return people


# =====================================================
# 5. SIMULATION DE L'ÉVACUATION
# =====================================================

def edge_key(a, b):

    return tuple(sorted((a, b)))


def simulate(graphs, people, max_time):

    t = 0.0

    next_entry = defaultdict(float)

    history = []
    building_history = []

    total_people = len(people)

    while t <= max_time:

        evacuated = sum(p["done"] for p in people)

        history.append({
            "time": t,
            "evacuated": evacuated,
            "inside": total_people - evacuated
        })

        building_row = {"time": t}

        for b in range(1, N_BUILDINGS + 1):

            building_row[f"Bâtiment {b}"] = sum(
                p["done"] and p["building"] == b
                for p in people
            )

        building_history.append(building_row)

        if evacuated == total_people:
            break

        # ---------------------------------------------
        # A. Mise à jour des personnes déjà en mouvement
        # ---------------------------------------------

        for p in people:

            if p["done"] or not p["moving"]:
                continue

            p["remaining"] -= DT

            if p["remaining"] <= 0:

                p["moving"] = False
                p["edge"] = None
                p["index"] += 1

                if p["index"] >= len(p["path"]) - 1:

                    p["done"] = True
                    p["evacuation_time"] = t

        # ---------------------------------------------
        # B. Comptage des personnes sur chaque liaison
        # ---------------------------------------------

        active = defaultdict(int)

        for p in people:

            if p["moving"] and p["edge"] is not None:
                active[p["edge"]] += 1

        # ---------------------------------------------
        # C. Entrée sur les liaisons et gestion des files
        # ---------------------------------------------

        waiting_people = sorted(
            [
                p for p in people
                if not p["done"]
                and not p["moving"]
                and t >= p["ready_time"]
                and p["index"] < len(p["path"]) - 1
            ],
            key=lambda p: p["ready_time"]
        )

        for p in waiting_people:

            G = graphs[p["building"]]

            a = p["path"][p["index"]]
            b = p["path"][p["index"] + 1]

            key = edge_key(a, b)

            edge_data = G[a][b]

            capacity = edge_data["capacity"]
            headway = edge_data["headway"]

            if (
                active[key] < capacity
                and t >= next_entry[key]
            ):

                distance = edge_data["length"]

                travel_time = distance / p["speed"]

                p["remaining"] = travel_time
                p["moving"] = True
                p["edge"] = key

                active[key] += 1

                next_entry[key] = t + headway

            else:

                p["waiting_time"] += DT

        t += DT

    return (
        pd.DataFrame(history),
        pd.DataFrame(building_history),
        people,
        t
    )


# =====================================================
# 6. AFFICHAGE DU GRAPHE DES BÂTIMENTS
# =====================================================

def draw_buildings(graphs):

    fig, axes = plt.subplots(
        1, 3,
        figsize=(17, 7)
    )

    colors = {
        "room": "#72B7E5",
        "corridor": "#F6C85F",
        "stair": "#9B8AE6",
        "exit": "#E45756"
    }

    for index, (building, G) in enumerate(graphs.items()):

        ax = axes[index]

        pos = nx.spring_layout(
            G,
            seed=building,
            k=0.8
        )

        node_colors = [
            colors[G.nodes[n]["kind"]]
            for n in G.nodes()
        ]

        nx.draw_networkx(
            G,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=180,
            font_size=5,
            with_labels=False,
            edge_color="#999999"
        )

        ax.set_title(f"Bâtiment {building}")
        ax.axis("off")

    plt.tight_layout()

    return fig


# =====================================================
# 7. INTERFACE STREAMLIT
# =====================================================

st.sidebar.header("⚙️ Paramètres de la simulation")

number_people = st.sidebar.slider(
    "Nombre de personnes",
    min_value=30,
    max_value=300,
    value=150,
    step=10
)

reaction_min = st.sidebar.slider(
    "Temps de réaction minimum (secondes)",
    0,
    30,
    5
)

reaction_max = st.sidebar.slider(
    "Temps de réaction maximum (secondes)",
    10,
    60,
    35
)

speed_mean = st.sidebar.slider(
    "Vitesse moyenne (m/s)",
    0.6,
    2.0,
    1.2,
    0.1
)

max_time = st.sidebar.slider(
    "Durée maximale de simulation (secondes)",
    300,
    3600,
    1800,
    100
)

seed = st.sidebar.number_input(
    "Graine aléatoire",
    min_value=1,
    max_value=10000,
    value=42
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "La structure des bâtiments reste fixe : "
    "3 bâtiments, 5 étages, 4 salles par étage, "
    "2 escaliers et 2 sorties par bâtiment."
)

# =====================================================
# 8. LANCEMENT
# =====================================================

if reaction_min >= reaction_max:

    st.error(
        "Le temps de réaction minimum doit être "
        "inférieur au temps maximum."
    )

else:

    if st.button("🚨 LANCER LA SIMULATION", type="primary"):

        with st.spinner("Simulation de l'évacuation en cours..."):

            graphs = build_all_graphs()

            # On adapte les temps de réaction et la vitesse
            people = generate_people(
                graphs,
                number_people,
                int(seed)
            )

            for p in people:

                p["reaction"] = random.uniform(
                    reaction_min,
                    reaction_max
                )

                p["ready_time"] = p["reaction"]

                p["speed"] = float(
                    np.clip(
                        np.random.normal(speed_mean, 0.20),
                        0.55,
                        2.0
                    )
                )

            progress = st.progress(0)

            history, building_history, people, final_time = simulate(
                graphs,
                people,
                max_time
            )

            progress.progress(100)

        st.success("Simulation terminée !")

        # ---------------------------------------------
        # 9. INDICATEURS PRINCIPAUX
        # ---------------------------------------------

        evacuated = sum(p["done"] for p in people)
        not_evacuated = number_people - evacuated

        evacuation_times = [
            p["evacuation_time"]
            for p in people
            if p["evacuation_time"] is not None
        ]

        waiting_times = [
            p["waiting_time"]
            for p in people
        ]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Personnes évacuées",
            f"{evacuated}/{number_people}"
        )

        col2.metric(
            "Personnes restantes",
            not_evacuated
        )

        col3.metric(
            "Durée de simulation",
            f"{final_time:.1f} s"
        )

        mean_wait = (
            np.mean(waiting_times)
            if waiting_times else 0
        )

        col4.metric(
            "Attente moyenne",
            f"{mean_wait:.1f} s"
        )

        if not_evacuated > 0:
            st.warning(
                "La durée maximale a été atteinte avant "
                "l'évacuation de toutes les personnes."
            )

        # ---------------------------------------------
        # 10. GRAPHE DE L'ÉVACUATION
        # ---------------------------------------------

        st.subheader("📈 Évolution de l'évacuation")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            history["time"],
            history["evacuated"],
            linewidth=2.5,
            label="Personnes évacuées"
        )

        ax.plot(
            history["time"],
            history["inside"],
            linewidth=2.5,
            label="Personnes encore dans les bâtiments"
        )

        ax.set_xlabel("Temps (secondes)")
        ax.set_ylabel("Nombre de personnes")
        ax.set_title("Progression de l'évacuation")
        ax.grid(alpha=0.3)
        ax.legend()

        st.pyplot(fig)

        # ---------------------------------------------
        # 11. COMPARAISON DES TROIS BÂTIMENTS
        # ---------------------------------------------

        st.subheader("🏢 Comparaison entre les bâtiments")

        fig, ax = plt.subplots(figsize=(10, 5))

        for b in range(1, N_BUILDINGS + 1):

            ax.plot(
                building_history["time"],
                building_history[f"Bâtiment {b}"],
                linewidth=2,
                label=f"Bâtiment {b}"
            )

        ax.set_xlabel("Temps (secondes)")
        ax.set_ylabel("Personnes évacuées")
        ax.set_title("Évacuation par bâtiment")
        ax.grid(alpha=0.3)
        ax.legend()

        st.pyplot(fig)

        # ---------------------------------------------
        # 12. DISTRIBUTION DES TEMPS D'ÉVACUATION
        # ---------------------------------------------

        st.subheader("📊 Distribution des temps d'évacuation")

        if evacuation_times:

            fig, ax = plt.subplots(figsize=(9, 4))

            ax.hist(
                evacuation_times,
                bins=15,
                edgecolor="black"
            )

            ax.set_xlabel("Temps d'évacuation (secondes)")
            ax.set_ylabel("Nombre de personnes")
            ax.set_title("Histogramme des temps d'évacuation")
            ax.grid(alpha=0.2)

            st.pyplot(fig)

            st.write(
                f"**Temps moyen individuel :** "
                f"{np.mean(evacuation_times):.2f} secondes"
            )

            st.write(
                f"**Écart-type :** "
                f"{np.std(evacuation_times):.2f} secondes"
            )

        # ---------------------------------------------
        # 13. DISTRIBUTION DES TEMPS D'ATTENTE
        # ---------------------------------------------

        st.subheader("⏳ Analyse des temps d'attente")

        fig, ax = plt.subplots(figsize=(9, 4))

        ax.hist(
            waiting_times,
            bins=15,
            edgecolor="black"
        )

        ax.set_xlabel("Temps d'attente (secondes)")
        ax.set_ylabel("Nombre de personnes")
        ax.set_title("Distribution des temps d'attente")
        ax.grid(alpha=0.2)

        st.pyplot(fig)

        # ---------------------------------------------
        # 14. STRUCTURE DES BÂTIMENTS
        # ---------------------------------------------

        st.subheader("🗺️ Architecture des bâtiments")

        st.pyplot(draw_buildings(graphs))

        # ---------------------------------------------
        # 15. TABLEAU DES RÉSULTATS
        # ---------------------------------------------

        st.subheader("📋 Résultats individuels")

        results = pd.DataFrame([
            {
                "Personne": p["id"] + 1,
                "Bâtiment": p["building"],
                "Vitesse (m/s)": round(p["speed"], 2),
                "Réaction (s)": round(p["reaction"], 2),
                "Attente (s)": round(p["waiting_time"], 2),
                "Évacuée": "Oui" if p["done"] else "Non",
                "Temps d'évacuation (s)": (
                    round(p["evacuation_time"], 2)
                    if p["evacuation_time"] is not None
                    else None
                )
            }
            for p in people
        ])

        st.dataframe(
            results,
            use_container_width=True
        )

        csv = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Télécharger les résultats en CSV",
            data=csv,
            file_name="resultats_evacuation.csv",
            mime="text/csv"
        )

        # ---------------------------------------------
        # 16. CONCLUSION AUTOMATIQUE
        # ---------------------------------------------

        st.subheader("📝 Interprétation des résultats")

        percentage = 100 * evacuated / number_people

        st.write(
            f"Sur les {number_people} personnes présentes, "
            f"{evacuated} ont été évacuées, soit "
            f"{percentage:.1f} %."
        )

        st.write(
            f"Le temps moyen d'attente est de "
            f"{mean_wait:.2f} secondes."
        )

        if evacuated == number_people:

            st.success(
                "Toutes les personnes ont été évacuées "
                "dans la durée de simulation."
            )

        else:

            st.warning(
                "Certaines personnes n'ont pas pu être évacuées "
                "dans la durée maximale choisie."
            )

        st.caption(
            "Modèle pédagogique fondé sur des hypothèses simplifiées. "
            "Les capacités, distances et vitesses doivent être "
            "calibrées avant toute interprétation réelle."
        )

st.markdown("---")
st.caption(
    "Projet de systèmes stochastiques — Simulation d'évacuation"
)
