import requests
import xml.etree.ElementTree as ET
import re


FEEDS = {
    "Atlantic": "https://www.nhc.noaa.gov/index-at.xml",
    "Eastern Pacific": "https://www.nhc.noaa.gov/index-ep.xml",
    "Central Pacific": "https://www.nhc.noaa.gov/index-cp.xml",
}


def get_live_cyclones():
    cyclones = []

    for basin, url in FEEDS.items():

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            root = ET.fromstring(response.content)

            for item in root.findall(".//item"):

                title = item.findtext("title", "")
                description = item.findtext("description", "")

                if not title.startswith("Summary for"):
                    continue

                # Storm name
                match = re.search(
                    r"Summary for (?:Tropical Storm|Hurricane|Tropical Depression|Post-Tropical Cyclone)\s+(.+?)\s+\(",
                    title,
                    re.IGNORECASE
                )

                name = match.group(1) if match else title

                # Coordinates
                location = re.search(
                    r"located near\s+([\d.-]+),\s*([\d.-]+)",
                    description
                )

                latitude = float(location.group(1)) if location else None
                longitude = float(location.group(2)) if location else None

                # Movement
                movement = re.search(
                    r"movement\s+([A-Za-z-]+)\s+at\s+([\d.]+)\s*mph",
                    description,
                    re.IGNORECASE
                )

                direction = movement.group(1) if movement else None
                speed_mph = float(movement.group(2)) if movement else None

                # Pressure
                pressure = re.search(
                    r"pressure was\s+(\d+)\s*mb",
                    description,
                    re.IGNORECASE
                )

                pressure_mb = float(pressure.group(1)) if pressure else None

                # Wind
                wind = re.search(
                    r"maximum sustained winds of about\s+(\d+)\s*mph",
                    description,
                    re.IGNORECASE
                )

                wind_mph = float(wind.group(1)) if wind else None

                cyclone = {
                    "name": name.strip(),
                    "basin": basin,
                    "latitude": latitude,
                    "longitude": longitude,
                    "wind_mph": wind_mph,
                    "pressure_mb": pressure_mb,
                    "movement_direction": direction,
                    "movement_speed_mph": speed_mph,
                    "source": "NOAA/NHC",
                }

                cyclones.append(cyclone)

        except Exception as e:
            print(f"Error reading {basin}: {e}")

    return cyclones


if __name__ == "__main__":

    cyclones = get_live_cyclones()

    print("\nLIVE CYCLONE DATA")
    print("=" * 50)

    if not cyclones:
        print("No active cyclones found.")

    for cyclone in cyclones:
        print(f"\nStorm: {cyclone['name']}")
        print(f"Basin: {cyclone['basin']}")
        print(f"Location: {cyclone['latitude']}°, {cyclone['longitude']}°")
        print(f"Wind: {cyclone['wind_mph']} mph")
        print(f"Pressure: {cyclone['pressure_mb']} mb")
        print(
            f"Movement: {cyclone['movement_direction']} "
            f"at {cyclone['movement_speed_mph']} mph"
        )
        print(f"Source: {cyclone['source']}")