import { useMemo } from 'react';
import * as THREE from 'three';
import type { HittingPointsData } from '../../types';
import { useAppStore } from '../../stores/useAppStore';

interface HittingPointsProps {
  data: HittingPointsData;
}

export function HittingPoints({ data }: HittingPointsProps) {
  const { settings } = useAppStore();

  const sphereGeometry = useMemo(() => new THREE.SphereGeometry(0.005, 16, 16), []);
  const sphereMaterial = useMemo(
    () => new THREE.MeshBasicMaterial({ color: 0x00ff00 }),
    []
  );

  if (!settings.showHittingPoints) {
    return null;
  }

  return (
    <group>
      {data.hitting_points.map((point, index) => (
        <mesh
          key={index}
          geometry={sphereGeometry}
          material={sphereMaterial}
          position={[point[0], point[1], point[2]]}
        />
      ))}
    </group>
  );
}

