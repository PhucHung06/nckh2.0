import os
import subprocess
import xml.etree.ElementTree as ET


class SumoEnvironment:
    def __init__(self, sumocfg_path, time_light_path, output_data_path):
        """
        Manage SUMO simulation runs and compute the GA fitness score.
        """
        self.sumocfg_path = sumocfg_path
        self.time_light_path = time_light_path
        self.output_data_path = output_data_path
        self.net_file_path = self._resolve_net_file_path()
        self.tl_id = "Center"

        # Fitness weights
        self.w_timeLoss = 0.35
        self.w_waitingTime = 0.35
        self.w_density = 0.15
        self.w_speed = 0.15

    def _resolve_net_file_path(self):
        """
        Read the net-file path from the SUMO config and resolve it to an absolute path.
        """
        tree = ET.parse(self.sumocfg_path)
        root = tree.getroot()
        net_file = root.find("./input/net-file")

        if net_file is None or not net_file.get("value"):
            raise ValueError("Could not find <net-file> in SUMO config")

        net_rel_path = net_file.get("value")
        return os.path.join(os.path.dirname(self.sumocfg_path), net_rel_path)

    def _build_time_light_xml(self, chromosome):
        g_ns, y_ns, g_ew, y_ew = chromosome
        return f"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<additional xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:noNamespaceSchemaLocation=\"http://sumo.dlr.de/xsd/additional_file.xsd\">\n    <tlLogic id=\"{self.tl_id}\" type=\"static\" programID=\"0\" offset=\"0\">\n        <phase duration=\"{g_ns}\" state=\"GGGGggrrrrrrGGGGggrrrrrr\"/>\n        <phase duration=\"{y_ns}\" state=\"yyyyyyrrrrrryyyyyyrrrrrr\"/>\n        <phase duration=\"{g_ew}\" state=\"rrrrrrGGGGggrrrrrrGGGGgg\"/>\n        <phase duration=\"{y_ew}\" state=\"rrrrrryyyyyyrrrrrryyyyyy\"/>\n    </tlLogic>\n</additional>\n"

    def write_time_light_xml(self, chromosome):
        """
        Save the chromosome to time_light.xml and update the active tlLogic in ngatu.net.xml.
        """
        xml_content = self._build_time_light_xml(chromosome)
        with open(self.time_light_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(xml_content)

        g_ns, y_ns, g_ew, y_ew = chromosome
        tree = ET.parse(self.net_file_path)
        root = tree.getroot()
        tl_logic = root.find(f".//tlLogic[@id='{self.tl_id}']")

        if tl_logic is None:
            raise ValueError(f"Traffic light '{self.tl_id}' was not found in {self.net_file_path}")

        phases = tl_logic.findall("phase")
        if len(phases) != 4:
            raise ValueError(
                f"Traffic light '{self.tl_id}' must have 4 phases, found {len(phases)}"
            )

        durations = [g_ns, y_ns, g_ew, y_ew]
        for phase, duration in zip(phases, durations):
            phase.set("duration", str(duration))

        tree.write(self.net_file_path, encoding="UTF-8", xml_declaration=True)

    def run_simulation(self):
        """
        Run SUMO in command-line mode for faster evaluation.
        """
        if os.path.exists(self.output_data_path):
            os.remove(self.output_data_path)

        command = [
            "sumo",
            "-c", self.sumocfg_path,
            "--no-step-log", "true",
            "--no-warnings", "true",
        ]

        data_dir = os.path.dirname(self.sumocfg_path)
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            cwd=data_dir,
            text=True,
        )

        if result.returncode != 0:
            print(f"\nSUMO error: {result.stderr}")

    def parse_output_and_calculate_fitness(self):
        """
        Read dulieu_matdo.xml (edgeData) and compute the fitness score.
        """
        if not os.path.exists(self.output_data_path):
            return -999999

        tree = ET.parse(self.output_data_path)
        root = tree.getroot()

        total_timeLoss = 0.0
        total_waitingTime = 0.0
        total_density = 0.0
        total_speed = 0.0
        count = 0

        for interval in root.findall('interval'):
            for edge in interval.findall('edge'):
                total_timeLoss += float(edge.get('timeLoss', 0))
                total_waitingTime += float(edge.get('waitingTime', 0))
                total_density += float(edge.get('density', 0))
                total_speed += float(edge.get('speed', 0))
                count += 1

        if count == 0:
            return -999999

        avg_tl = total_timeLoss / count
        avg_wt = total_waitingTime / count
        avg_den = total_density / count
        avg_spd = total_speed / count

        fitness = (
            self.w_speed * avg_spd
            - self.w_timeLoss * avg_tl
            - self.w_waitingTime * avg_wt
            - self.w_density * avg_den
        )

        return fitness

    def evaluate(self, chromosome):
        """
        Full evaluation pipeline: update traffic light -> run SUMO -> parse output.
        """
        try:
            self.write_time_light_xml(chromosome)
            self.run_simulation()
            return self.parse_output_and_calculate_fitness()
        except Exception as exc:
            print(f"\nEvaluation error for {chromosome}: {exc}")
            return -999999
