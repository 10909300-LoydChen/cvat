// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { useSyncExternalStore } from 'react';

// Client of the active learning controller (components/online-learning/controller),
// served on the CVAT host under /al/ and authorized with the CVAT session cookie

export interface ModelStatus {
    version: number;
    labels: string[];
    metrics: { round?: string; val_map50?: number; val_samples?: number };
    samples: number;
    pending_new_samples: number;
    full_round_requested: boolean;
    paused: boolean;
    training: boolean;
}

export interface FrameScore {
    frame: number;
    score: number;
    detections: number;
    annotated: boolean;
}

export interface JobScores {
    state: 'idle' | 'running' | 'done' | 'failed';
    progress: number;
    total: number;
    model_version: number | null;
    frames: FrameScore[];
    error?: string;
}

async function request<T>(path: string, org: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`/al/${path}?org=${encodeURIComponent(org)}`, {
        credentials: 'same-origin',
        ...init,
        headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    });
    if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail ?? `${response.status} ${response.statusText}`);
    }
    return response.json();
}

export function fetchModelStatus(org: string): Promise<ModelStatus> {
    return request<ModelStatus>('status', org);
}

export function controlTraining(action: 'pause' | 'resume' | 'train', org: string): Promise<ModelStatus> {
    return request<ModelStatus>('training', org, { method: 'POST', body: JSON.stringify({ action }) });
}

const scores: Record<number, JobScores> = {};
const listeners = new Set<() => void>();

function setScores(jobID: number, value: JobScores): void {
    scores[jobID] = value;
    listeners.forEach((listener) => listener());
}

async function pollScores(jobID: number, org: string): Promise<void> {
    let value = await request<JobScores>(`jobs/${jobID}/scores`, org);
    setScores(jobID, value);
    while (value.state === 'running') {
        await new Promise((resolve) => { setTimeout(resolve, 1000); });
        value = await request<JobScores>(`jobs/${jobID}/scores`, org);
        setScores(jobID, value);
    }
}

export async function scoreJob(jobID: number, org: string): Promise<void> {
    setScores(jobID, await request<JobScores>(`jobs/${jobID}/scores`, org, { method: 'POST' }));
    await pollScores(jobID, org);
}

export function loadJobScores(jobID: number, org: string): Promise<void> {
    return pollScores(jobID, org);
}

export function useJobScores(jobID: number | undefined): JobScores | null {
    return useSyncExternalStore(
        (listener) => {
            listeners.add(listener);
            return () => listeners.delete(listener);
        },
        () => (typeof jobID === 'number' ? scores[jobID] ?? null : null),
    );
}

// The most uncertain frame that is not annotated yet, other than the current one
export function nextSuggestedFrame(value: JobScores | null, currentFrame: number): FrameScore | null {
    return (value?.frames ?? [])
        .filter((item) => !item.annotated && item.frame !== currentFrame && item.score > 0)
        .sort((a, b) => b.score - a.score)[0] ?? null;
}
