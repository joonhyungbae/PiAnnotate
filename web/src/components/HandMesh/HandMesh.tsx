import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useAppStore } from '../../stores/useAppStore';

interface HandMeshProps {
  hand: 'left' | 'right';
  faces: number[][];
}

const NUM_VERTICES = 778;

export function HandMesh({ hand, faces }: HandMeshProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const { currentFrameData, settings } = useAppStore();

  // Create geometry
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(NUM_VERTICES * 3);
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setIndex(faces.flat());
    return geo;
  }, [faces]);

  // Create material
  const material = useMemo(() => {
    return new THREE.MeshStandardMaterial({
      color: settings.handColor,
      roughness: 1.0,
      metalness: 1.0,
    });
  }, [settings.handColor]);

  // Update vertices based on frame data
  useEffect(() => {
    if (!meshRef.current || !currentFrameData) return;

    const vertices =
      hand === 'left'
        ? currentFrameData.left_vertices
        : currentFrameData.right_vertices;

    if (!vertices || vertices.length === 0) return;

    const positions = geometry.attributes.position as THREE.BufferAttribute;
    const posArray = positions.array as Float32Array;

    for (let i = 0; i < NUM_VERTICES && i < vertices.length; i++) {
      for (let j = 0; j < 3; j++) {
        posArray[i * 3 + j] = parseFloat(vertices[i][j]);
      }
    }

    geometry.computeVertexNormals();
    positions.needsUpdate = true;
  }, [currentFrameData, hand, geometry]);

  return (
    <mesh ref={meshRef} geometry={geometry} material={material} frustumCulled={false} />
  );
}

