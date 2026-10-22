// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useEffect, useRef, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import Text from 'antd/lib/typography/Text';

import {
    Label, ObjectState, ObjectType, ShapeType,
} from 'cvat-core-wrapper';
import { CombinedState } from 'reducers';
import { shallowEqual, ThunkDispatch } from 'utils/redux';
import { activateObject, updateAnnotationsAsync } from 'actions/annotation-actions';
import { filterApplicableLabels } from 'utils/filter-applicable-labels';
import LabelSelector from 'components/label-selector/label-selector';
import ItemAttribute from './object-item-attribute';

const THUMBNAIL_MAX_WIDTH = 288;
const THUMBNAIL_MAX_HEIGHT = 96;
const CONTEXT_PADDING = 4; // frame pixels around the object

interface FrameImage {
    bitmap: ImageBitmap;
    width: number; // frame size, the coordinate space of object points
    height: number;
}

type Box = [number, number, number, number];

function getBoundingBox(state: ObjectState): Box | null {
    const points = state.points ?? [];
    if (!points.length) {
        return null;
    }

    if (state.shapeType === ShapeType.MASK) {
        // mask points are RLE data followed by left, top, right, bottom
        const [left, top, right, bottom] = points.slice(-4);
        return [left, top, right + 1, bottom + 1];
    }

    if (state.shapeType === ShapeType.ELLIPSE) {
        const [cx, cy, rightX, topY] = points;
        const rx = Math.abs(rightX - cx);
        const ry = Math.abs(cy - topY);
        return [cx - rx, cy - ry, cx + rx, cy + ry];
    }

    const xs = points.filter((_: number, idx: number) => idx % 2 === 0);
    const ys = points.filter((_: number, idx: number) => idx % 2 === 1);
    return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

function ObjectThumbnail({ image, box }: { image: FrameImage | null; box: Box }): JSX.Element {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !image) {
            return;
        }

        const x1 = Math.max(0, box[0] - CONTEXT_PADDING);
        const y1 = Math.max(0, box[1] - CONTEXT_PADDING);
        const x2 = Math.min(image.width, box[2] + CONTEXT_PADDING);
        const y2 = Math.min(image.height, box[3] + CONTEXT_PADDING);
        const width = Math.max(1, x2 - x1);
        const height = Math.max(1, y2 - y1);
        const scale = Math.min(THUMBNAIL_MAX_HEIGHT / height, THUMBNAIL_MAX_WIDTH / width);
        canvas.width = Math.max(1, Math.round(width * scale));
        canvas.height = Math.max(1, Math.round(height * scale));

        // the decoded frame can have another resolution than the frame (e.g. compressed chunks)
        const sx = image.bitmap.width / image.width;
        const sy = image.bitmap.height / image.height;
        try {
            canvas.getContext('2d')?.drawImage(
                image.bitmap, x1 * sx, y1 * sy, width * sx, height * sy, 0, 0, canvas.width, canvas.height,
            );
        } catch (_: unknown) {
            // the bitmap can be released by the frame cache while switching frames
        }
    }, [image, ...box]);

    return <canvas ref={canvasRef} className='cvat-annotated-object-thumbnail-canvas' />;
}

interface CardProps {
    objectState: ObjectState;
    box: Box;
    image: FrameImage | null;
    labels: Label[];
    active: boolean;
}

function AnnotatedObjectCard(props: CardProps): JSX.Element {
    const {
        objectState, box, image, labels, active,
    } = props;
    const dispatch = useDispatch<ThunkDispatch>();
    const readonly = objectState.lock || objectState.isGroundTruth;

    const changeAttribute = (attrID: number, value: string): void => {
        objectState.attributes = { [attrID]: value };
        dispatch(updateAnnotationsAsync([objectState]));
    };

    const changeLabel = (label: Label): void => {
        objectState.label = label;
        dispatch(updateAnnotationsAsync([objectState]));
    };

    return (
        <div
            className={`cvat-annotated-object-card${active ? ' cvat-annotated-object-card-active' : ''}`}
            style={{ borderColor: objectState.label.color }}
        >
            <button
                type='button'
                className='cvat-annotated-object-thumbnail'
                onClick={() => dispatch(activateObject(objectState.clientID as number, null, null))}
            >
                <ObjectThumbnail image={image} box={box} />
            </button>
            <div className='cvat-annotated-object-card-details'>
                <div className='cvat-annotated-object-card-header'>
                    <Text type='secondary'>{objectState.clientID}</Text>
                    <LabelSelector
                        size='small'
                        labels={filterApplicableLabels(objectState, labels)}
                        value={objectState.label}
                        onChange={changeLabel}
                        disabled={readonly}
                        popupMatchSelectWidth={false}
                    />
                </div>
                <div className='cvat-annotated-object-card-attributes'>
                    {objectState.label.attributes.map((attribute) => (
                        <ItemAttribute
                            key={attribute.id}
                            readonly={readonly}
                            attrValue={objectState.attributes[attribute.id as number]}
                            attrInputType={attribute.inputType}
                            attrName={attribute.name}
                            attrID={attribute.id as number}
                            attrValues={attribute.values}
                            changeAttribute={changeAttribute}
                        />
                    ))}
                </div>
            </div>
        </div>
    );
}

function ObjectAttributesList(): JSX.Element {
    const {
        states, frameData, labels, activatedStateID,
    } = useSelector((state: CombinedState) => ({
        states: state.annotation.annotations.states,
        frameData: state.annotation.player.frame.data,
        labels: state.annotation.job.labels,
        activatedStateID: state.annotation.annotations.activatedStateID,
    }), shallowEqual);
    const [image, setImage] = useState<FrameImage | null>(null);

    useEffect(() => {
        let cancelled = false;
        setImage(null);
        if (frameData) {
            frameData.data().then(async ({ imageData, renderWidth, renderHeight }: {
                imageData: ImageBitmap | Blob; renderWidth: number; renderHeight: number;
            }) => {
                const bitmap = imageData instanceof Blob ? await createImageBitmap(imageData) : imageData;
                if (!cancelled) {
                    setImage({ bitmap, width: renderWidth, height: renderHeight });
                }
            }).catch(() => {});
        }
        return () => { cancelled = true; };
    }, [frameData]);

    const objects = states
        .filter((objectState: ObjectState) => (
            [ObjectType.SHAPE, ObjectType.TRACK].includes(objectState.objectType) &&
            !objectState.outside && !objectState.hidden
        ))
        .map((objectState: ObjectState) => ({ objectState, box: getBoundingBox(objectState) }))
        .filter((item): item is { objectState: ObjectState; box: Box } => item.box !== null);

    return (
        <div className='cvat-object-attributes-list'>
            <div className='cvat-object-attributes-list-header'>
                <Text strong>{`Objects on this frame: ${objects.length}`}</Text>
            </div>
            <div className='cvat-object-attributes-list-items'>
                {objects.length ? objects.map(({ objectState, box }) => (
                    <AnnotatedObjectCard
                        key={objectState.clientID}
                        objectState={objectState}
                        box={box}
                        image={image}
                        labels={labels}
                        active={objectState.clientID === activatedStateID}
                    />
                )) : <Text type='secondary'>No annotated objects on this frame</Text>}
            </div>
        </div>
    );
}

export default React.memo(ObjectAttributesList);
