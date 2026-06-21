import os
import shutil
from agents.data_steward_agent import DataStewardAgent
from agents.scientist_agent import ScientistAgent
from agents.architect_agent import ArchitectAgent
from agents.inspector_agent import InspectorAgent

def main():
    state_dir = ".system_states"
    if os.path.exists(state_dir):
        shutil.rmtree(state_dir)
    os.makedirs(state_dir, exist_ok=True)
    print("Nettoyage et initialisation des repertoires de controle...")
    steward = DataStewardAgent()
    scientist = ScientistAgent()
    architect = ArchitectAgent()
    inspector = InspectorAgent()
    start_signal = os.path.join(state_dir, "start_pipeline")
    print(f"Creation du signal d initialisation : {start_signal}")
    with open(start_signal, "w") as f:
        f.write("trigger")
    print("\n--- Lancement du DataStewardAgent ---")
    steward.run()
    print("DataStewardAgent a termine son traitement.")
    print("\n--- Lancement du ScientistAgent ---")
    scientist.run()
    print("ScientistAgent a termine son traitement.")
    print("\n--- Lancement du ArchitectAgent ---")
    architect.run()
    print("ArchitectAgent a termine son traitement.")
    print("\n--- Lancement du InspectorAgent ---")
    inspector.run()
    print("InspectorAgent a termine sa simulation.")
    print("\nPipeline complet execute avec succes.")

if __name__ == "__main__":
    main()