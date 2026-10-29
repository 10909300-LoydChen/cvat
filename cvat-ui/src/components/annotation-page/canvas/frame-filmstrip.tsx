// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useEffect, useRef, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import Button from 'antd/lib/button';
import Checkbox from 'antd/lib/checkbox';
import Text from 'antd/lib/typography/Text';
import { DownOutlined, ExportOutlined, UpOutlined } from '@ant-design/icons';

import { getCore } from 'cvat-core-wrapper';
import { CombinedState } from 'reducers';
import { shallowEqual, ThunkDispatch } from 'utils/redux';
import { changeFrameAsync } from 'actions/annotation-actions';
import { exportActions } from 'actions/export-actions';
import { FrameScore, useJobScores } from 'utils/active-learning';

const core = getCore();

// the selection is a per-browser convenience, so it is kept in localStorage
const selectionKey = (jobID: number): string => `cvat.filmstrip.selection.${jobID}`;

function loadSelection(jobID: number): Set<number> {
    try {
        return new Set(JSON.parse(localStorage.getItem(selectionKey(jobID)) ?? '[]'));
    } catch {
        return new Set();
    }
}

function saveSelection(jobID: number, selection: Set<number>): void {
    try {
        localStorage.setItem(selectionKey(jobID), JSON.stringify([...selection]));
    } catch {
        // storage is not available, the selection lives until the page is reloaded
    }
}

function FrameFilmstrip(): JSX.Element | null {
    const dispatch = useDispatch<ThunkDispatch>();
    const {
        job, frameNumbers, deletedFrames, currentFrame, organization, navigationBlocked,
    } = useSelector((state: CombinedState) => ({
        job: state.annotation.job.instance,
        frameNumbers: state.annotation.job.frameNumbers,
        deletedFrames: state.annotation.job.meta?.deletedFrames ?? {},
        currentFrame: state.annotation.player.frame.number,
        organization: state.organizations.current,
        navigationBlocked: state.annotation.player.navigationBlocked,
    }), shallowEqual);
    const [collapsed, setCollapsed] = useState(false);
    const [selection, setSelection] = useState<Set<number>>(new Set());
    const lastToggledRef = useRef<number | null>(null);
    const itemsRef = useRef<HTMLDivElement>(null);
    const scores = useJobScores(job?.id);
    const scoreByFrame = Object.fromEntries(
        (scores?.frames ?? []).map((item: FrameScore) => [item.frame, item]),
    ) as Record<number, FrameScore>;

    useEffect(() => {
        // the canvas layout subtracts the filmstrip height when it computes its own size
        window.dispatchEvent(new Event('resize'));
    }, [collapsed]);

    useEffect(() => {
        if (job) {
            setSelection(loadSelection(job.id));
        }
    }, [job?.id]);

    useEffect(() => {
        itemsRef.current
            ?.querySelector(`[data-frame="${currentFrame}"]`)
            ?.scrollIntoView({ block: 'nearest', inline: 'center' });
    }, [currentFrame, collapsed]);

    if (!job) {
        return null;
    }

    // frames are loaded by the browser only when they scroll into view; the session cookie authorizes them
    const frameURL = (frame: number): string => (
        `${core.config.backendAPI}/jobs/${job.id}/data?type=frame&quality=compressed` +
        `&number=${frame}&org=${organization?.slug ?? ''}`
    );
    const position = frameNumbers.indexOf(currentFrame) + 1;
    const selected = frameNumbers.filter((frame: number) => selection.has(frame));

    const updateSelection = (next: Set<number>): void => {
        setSelection(next);
        saveSelection(job.id, next);
    };

    // shift+click selects or clears the whole range from the previously toggled frame
    const toggle = (frame: number, checked: boolean, range: boolean): void => {
        const next = new Set(selection);
        const last = lastToggledRef.current;
        const frames = range && last !== null ? frameNumbers.filter((item: number) => (
            item >= Math.min(last, frame) && item <= Math.max(last, frame)
        )) : [frame];
        frames.forEach((item: number) => (checked ? next.add(item) : next.delete(item)));
        lastToggledRef.current = frame;
        updateSelection(next);
    };

    return (
        <div className='cvat-frame-filmstrip'>
            <div className='cvat-frame-filmstrip-header'>
                <Text strong>{`Frames ${position} / ${frameNumbers.length}`}</Text>
                <div className='cvat-frame-filmstrip-actions'>
                    <Text type='secondary'>{`${selected.length} selected`}</Text>
                    <Button
                        size='small'
                        type='link'
                        disabled={selected.length === frameNumbers.length}
                        onClick={() => updateSelection(new Set(frameNumbers))}
                    >
                        Select all
                    </Button>
                    <Button
                        size='small'
                        type='link'
                        disabled={!selected.length}
                        onClick={() => updateSelection(new Set())}
                    >
                        Clear
                    </Button>
                    <Button
                        size='small'
                        className='cvat-frame-filmstrip-export'
                        icon={<ExportOutlined />}
                        disabled={!selected.length}
                        onClick={() => dispatch(exportActions.openExportDatasetModal(job, selected))}
                    >
                        Export selected
                    </Button>
                    <button
                        type='button'
                        className='cvat-frame-filmstrip-toggle'
                        aria-label={collapsed ? 'Show frames' : 'Hide frames'}
                        onClick={() => setCollapsed(!collapsed)}
                    >
                        {collapsed ? <UpOutlined /> : <DownOutlined />}
                    </button>
                </div>
            </div>
            {!collapsed && (
                <div className='cvat-frame-filmstrip-items' ref={itemsRef}>
                    {frameNumbers.map((frame: number) => (
                        <div
                            key={frame}
                            data-frame={frame}
                            className={[
                                'cvat-frame-filmstrip-item',
                                frame === currentFrame ? 'cvat-frame-filmstrip-item-current' : '',
                                deletedFrames[frame] ? 'cvat-frame-filmstrip-item-deleted' : '',
                                selection.has(frame) ? 'cvat-frame-filmstrip-item-checked' : '',
                            ].join(' ')}
                        >
                            <button
                                type='button'
                                className='cvat-frame-filmstrip-thumb'
                                disabled={navigationBlocked}
                                onClick={() => {
                                    if (frame !== currentFrame) {
                                        dispatch(changeFrameAsync(frame));
                                    }
                                }}
                            >
                                <img src={frameURL(frame)} loading='lazy' alt={`Frame ${frame}`} />
                                <span className='cvat-frame-filmstrip-number'>{frame}</span>
                                {scoreByFrame[frame] && (
                                    <span
                                        className={[
                                            'cvat-frame-filmstrip-score',
                                            scoreByFrame[frame].annotated ? 'cvat-frame-filmstrip-score-annotated' : '',
                                        ].join(' ')}
                                        title='Active learning score: how unsure the model is about this frame'
                                    >
                                        {scoreByFrame[frame].annotated ? '✓' : Math.round(scoreByFrame[frame].score * 100)}
                                    </span>
                                )}
                            </button>
                            <Checkbox
                                className='cvat-frame-filmstrip-check'
                                checked={selection.has(frame)}
                                aria-label={`Select frame ${frame} for export`}
                                onChange={(event) => toggle(
                                    frame,
                                    event.target.checked,
                                    (event.nativeEvent as MouseEvent).shiftKey,
                                )}
                            />
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

export default React.memo(FrameFilmstrip);
