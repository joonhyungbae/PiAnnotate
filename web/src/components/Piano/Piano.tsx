import { useEffect, useRef } from 'react';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader.js';
import { getPianoMeshUrl } from '../../api';
import { useAppStore } from '../../stores/useAppStore';

// Check if black key
function isBlackKey(keyIdx: number): boolean {
  const blackKeys = [1, 3, 6, 8, 10];
  return blackKeys.includes((keyIdx + 8) % 12);
}

interface PianoKeyProps {
  keyIndex: number;
}

function PianoKey({ keyIndex }: PianoKeyProps) {
  const meshRef = useRef<THREE.Group>(null);
  const materialRef = useRef<THREE.MeshStandardMaterial | null>(null);
  const { currentFrameData } = useAppStore();
  const { scene } = useThree();

  const isBlack = isBlackKey(keyIndex + 1);
  const originalColor = isBlack ? 0x000000 : 0xffffff;

  useEffect(() => {
    const mtlUrl = getPianoMeshUrl(keyIndex, 'mtl');
    const objUrl = getPianoMeshUrl(keyIndex, 'obj');

    const mtlLoader = new MTLLoader();

    mtlLoader.load(mtlUrl, (materials) => {
      materials.preload();

      const objLoader = new OBJLoader();
      objLoader.setMaterials(materials);

      objLoader.load(objUrl, (object) => {
        object.traverse((child) => {
          if (child instanceof THREE.Mesh) {
            const mat = new THREE.MeshStandardMaterial({
              color: originalColor,
              roughness: isBlack ? 0.4 : 0.3,
              metalness: isBlack ? 0.1 : 0.2,
            });
            child.material = mat;
            materialRef.current = mat;
            child.castShadow = true;
            child.receiveShadow = true;
          }
        });

        object.rotation.x = Math.PI / 2;

        if (meshRef.current) {
          meshRef.current.add(object);
        }
      });
    });

    return () => {
      if (meshRef.current) {
        meshRef.current.children.forEach((child) => {
          scene.remove(child);
        });
      }
    };
  }, [keyIndex, scene, isBlack, originalColor]);

  // Update key states based on frame data
  useEffect(() => {
    if (!meshRef.current || !materialRef.current) return;

    const isPressed = (currentFrameData?.pressed_keys?.[keyIndex] ?? 0) > 0;

    if (isPressed) {
      materialRef.current.color.setHex(0x00ff00);
      meshRef.current.rotation.z = -Math.PI / 45;
    } else {
      materialRef.current.color.setHex(originalColor);
      meshRef.current.rotation.z = 0;
    }

    materialRef.current.needsUpdate = true;
  }, [currentFrameData, keyIndex, originalColor]);

  return <group ref={meshRef} />;
}

export function Piano() {
  const keys = [];
  for (let i = 0; i < 88; i++) {
    keys.push(<PianoKey key={i} keyIndex={i} />);
  }
  return <group>{keys}</group>;
}

