import { useRef, useEffect } from 'react';
import * as THREE from 'three';
import { useAppStore } from '../../stores/useAppStore';

// Fingertip indices (MANO model)
const FINGER_TIP_INDICES: Record<string, number> = {
  thumb: 4,
  index: 8,
  middle: 12,
  ring: 16,
  pinky: 20,
};

// Create texture for finger number
function createFingerTexture(fingerNumber: number): THREE.Texture {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d')!;
  canvas.width = 64;
  canvas.height = 64;

  // Transparent background
  context.clearRect(0, 0, 64, 64);

  // Red text
  context.fillStyle = '#ff0000';
  context.font = 'bold 48px Arial';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(fingerNumber.toString(), 32, 32);

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

export function FingeringLabels() {
  const { currentFrameData, currentFingering, settings } = useAppStore();
  const groupRef = useRef<THREE.Group>(null);
  const spritesRef = useRef<Map<string, THREE.Sprite>>(new Map());
  const texturesRef = useRef<Map<number, THREE.Texture>>(new Map());

  // Get texture from cache
  const getTexture = (fingerNumber: number): THREE.Texture => {
    if (!texturesRef.current.has(fingerNumber)) {
      texturesRef.current.set(fingerNumber, createFingerTexture(fingerNumber));
    }
    return texturesRef.current.get(fingerNumber)!;
  };

  // Update fingering
  useEffect(() => {
    if (!groupRef.current || !settings.showFingering || !currentFrameData) {
      // Hide all sprites
      spritesRef.current.forEach((sprite) => {
        sprite.visible = false;
      });
      return;
    }

    const leftJoints = currentFrameData.left_joints?.map((v) => parseFloat(v)) || [];
    const rightJoints = currentFrameData.right_joints?.map((v) => parseFloat(v)) || [];

    // Hide all sprites
    spritesRef.current.forEach((sprite) => {
      sprite.visible = false;
    });

    // Show sprites based on current fingering
    currentFingering.forEach((fingering) => {
      const { hand, finger_name, finger } = fingering;
      const tipIndex = FINGER_TIP_INDICES[finger_name];
      if (tipIndex === undefined) return;

      const joints = hand === 'left' ? leftJoints : rightJoints;
      if (joints.length < (tipIndex + 1) * 3) return;

      const baseIndex = tipIndex * 3;
      const position = new THREE.Vector3(
        joints[baseIndex],
        joints[baseIndex + 1],
        joints[baseIndex + 2]
      );

      const spriteKey = `${hand}_${finger_name}`;
      let sprite = spritesRef.current.get(spriteKey);

      if (!sprite) {
        const texture = getTexture(finger);
        const material = new THREE.SpriteMaterial({
          map: texture,
          transparent: true,
          depthTest: false,
          depthWrite: false,
        });
        sprite = new THREE.Sprite(material);
        sprite.scale.set(0.04, 0.04, 1);
        sprite.renderOrder = 999;
        groupRef.current!.add(sprite);
        spritesRef.current.set(spriteKey, sprite);
      }

      // Update texture
      const currentTexture = getTexture(finger);
      if ((sprite.material as THREE.SpriteMaterial).map !== currentTexture) {
        (sprite.material as THREE.SpriteMaterial).map = currentTexture;
        (sprite.material as THREE.SpriteMaterial).needsUpdate = true;
      }

      sprite.position.copy(position);
      sprite.visible = true;
    });
  }, [currentFrameData, currentFingering, settings.showFingering]);

  return <group ref={groupRef} />;
}

