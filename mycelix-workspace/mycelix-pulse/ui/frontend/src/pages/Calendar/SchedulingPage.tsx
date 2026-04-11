// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Scheduling Page
 *
 * Meeting scheduling, availability management, and scheduling links
 */

import React, { useEffect, useState } from 'react';

interface SchedulingLink {
  id: string;
  slug: string;
  title: string;
  description?: string;
  duration_minutes: number;
  buffer_before: number;
  buffer_after: number;
  advance_notice_hours: number;
  max_days_ahead: number;
  available_days: number[];
  start_hour: number;
  end_hour: number;
  active: boolean;
  created_at: string;
}

interface AvailableSlot {
  start: string;
  end: string;
  score: number;
}

interface BookingFormData {
  name: string;
  email: string;
  notes: string;
  selectedSlot: AvailableSlot | null;
}

export default function SchedulingPage() {
  const [links, setLinks] = useState<SchedulingLink[]>([]);
  const [selectedLink, setSelectedLink] = useState<SchedulingLink | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSchedulingLinks();
  }, []);

  async function fetchSchedulingLinks() {
    try {
      const response = await fetch('/api/scheduling/links');
      if (response.ok) {
        setLinks(await response.json());
      }
    } catch (error) {
      console.error('Failed to fetch scheduling links:', error);
    } finally {
      setLoading(false);
    }
  }

  async function createLink(data: Partial<SchedulingLink>) {
    try {
      const response = await fetch('/api/scheduling/links', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      if (response.ok) {
        const newLink = await response.json();
        setLinks([...links, newLink]);
        setShowCreateModal(false);
      }
    } catch (error) {
      console.error('Failed to create scheduling link:', error);
    }
  }

  async function toggleLinkActive(link: SchedulingLink) {
    try {
      const response = await fetch(`/api/scheduling/links/${link.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ active: !link.active }),
      });
      if (response.ok) {
        setLinks(links.map(l => l.id === link.id ? { ...l, active: !l.active } : l));
      }
    } catch (error) {
      console.error('Failed to toggle link:', error);
    }
  }

  async function deleteLink(linkId: string) {
    if (!confirm('Delete this scheduling link?')) return;
    try {
      const response = await fetch(`/api/scheduling/links/${linkId}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        setLinks(links.filter(l => l.id !== linkId));
      }
    } catch (error) {
      console.error('Failed to delete link:', error);
    }
  }

  const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Scheduling Links</h1>
          <p className="text-muted">Create shareable links for others to book time with you</p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary/90"
        >
          Create Link
        </button>
      </div>

      {links.length === 0 ? (
        <div className="text-center py-12 bg-surface rounded-lg border border-border">
          <div className="text-4xl mb-4">📅</div>
          <h2 className="text-lg font-semibold mb-2">No scheduling links yet</h2>
          <p className="text-muted mb-4">Create a link to let others book time on your calendar</p>
          <button
            onClick={() => setShowCreateModal(true)}
            className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary/90"
          >
            Create Your First Link
          </button>
        </div>
      ) : (
        <div className="space-y-4">
          {links.map((link) => (
            <div
              key={link.id}
              className={`bg-surface rounded-lg border p-4 ${
                link.active ? 'border-border' : 'border-border/50 opacity-60'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <h3 className="font-semibold">{link.title}</h3>
                    <span className={`px-2 py-0.5 text-xs rounded-full ${
                      link.active
                        ? 'bg-green-100 text-green-800'
                        : 'bg-gray-100 text-gray-600'
                    }`}>
                      {link.active ? 'Active' : 'Inactive'}
                    </span>
                  </div>
                  {link.description && (
                    <p className="text-sm text-muted mt-1">{link.description}</p>
                  )}
                  <div className="flex items-center gap-4 mt-3 text-sm text-muted">
                    <span>{link.duration_minutes} min</span>
                    <span>
                      {link.available_days.map(d => dayNames[d]).join(', ')}
                    </span>
                    <span>
                      {formatTime(link.start_hour)} - {formatTime(link.end_hour)}
                    </span>
                  </div>
                  <div className="mt-3 flex items-center gap-2">
                    <input
                      type="text"
                      readOnly
                      value={`${window.location.origin}/schedule/${link.slug}`}
                      className="flex-1 px-3 py-1.5 bg-background border border-border rounded text-sm"
                    />
                    <button
                      onClick={() => {
                        navigator.clipboard.writeText(
                          `${window.location.origin}/schedule/${link.slug}`
                        );
                      }}
                      className="px-3 py-1.5 text-sm border border-border rounded hover:bg-muted/30"
                    >
                      Copy
                    </button>
                  </div>
                </div>
                <div className="flex items-center gap-2 ml-4">
                  <button
                    onClick={() => toggleLinkActive(link)}
                    className="p-2 hover:bg-muted/30 rounded"
                    title={link.active ? 'Deactivate' : 'Activate'}
                  >
                    {link.active ? '⏸️' : '▶️'}
                  </button>
                  <button
                    onClick={() => setSelectedLink(link)}
                    className="p-2 hover:bg-muted/30 rounded"
                    title="Edit"
                  >
                    ✏️
                  </button>
                  <button
                    onClick={() => deleteLink(link.id)}
                    className="p-2 hover:bg-red-50 text-red-600 rounded"
                    title="Delete"
                  >
                    🗑️
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Create/Edit Modal */}
      {(showCreateModal || selectedLink) && (
        <SchedulingLinkModal
          link={selectedLink}
          onSave={(data) => {
            if (selectedLink) {
              // Update existing
              fetch(`/api/scheduling/links/${selectedLink.id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
              }).then(() => {
                fetchSchedulingLinks();
                setSelectedLink(null);
              });
            } else {
              createLink(data);
            }
          }}
          onClose={() => {
            setShowCreateModal(false);
            setSelectedLink(null);
          }}
        />
      )}
    </div>
  );
}

function formatTime(hour: number): string {
  const h = hour % 12 || 12;
  const ampm = hour < 12 ? 'AM' : 'PM';
  return `${h}:00 ${ampm}`;
}

interface SchedulingLinkModalProps {
  link: SchedulingLink | null;
  onSave: (data: Partial<SchedulingLink>) => void;
  onClose: () => void;
}

function SchedulingLinkModal({ link, onSave, onClose }: SchedulingLinkModalProps) {
  const [formData, setFormData] = useState({
    title: link?.title || '',
    description: link?.description || '',
    slug: link?.slug || '',
    duration_minutes: link?.duration_minutes || 30,
    buffer_before: link?.buffer_before || 0,
    buffer_after: link?.buffer_after || 0,
    advance_notice_hours: link?.advance_notice_hours || 24,
    max_days_ahead: link?.max_days_ahead || 60,
    available_days: link?.available_days || [1, 2, 3, 4, 5],
    start_hour: link?.start_hour || 9,
    end_hour: link?.end_hour || 17,
  });

  const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

  function toggleDay(day: number) {
    setFormData(prev => ({
      ...prev,
      available_days: prev.available_days.includes(day)
        ? prev.available_days.filter(d => d !== day)
        : [...prev.available_days, day].sort(),
    }));
  }

  function generateSlug(title: string) {
    return title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '');
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-background rounded-lg shadow-xl w-full max-w-lg max-h-[90vh] overflow-y-auto">
        <div className="p-4 border-b border-border">
          <h2 className="text-lg font-semibold">
            {link ? 'Edit Scheduling Link' : 'Create Scheduling Link'}
          </h2>
        </div>

        <div className="p-4 space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Title</label>
            <input
              type="text"
              value={formData.title}
              onChange={(e) => {
                const title = e.target.value;
                setFormData(prev => ({
                  ...prev,
                  title,
                  slug: prev.slug || generateSlug(title),
                }));
              }}
              placeholder="30 Minute Meeting"
              className="w-full px-3 py-2 border border-border rounded-lg"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">URL Slug</label>
            <div className="flex items-center gap-2">
              <span className="text-muted text-sm">/schedule/</span>
              <input
                type="text"
                value={formData.slug}
                onChange={(e) => setFormData({ ...formData, slug: e.target.value })}
                placeholder="30min"
                className="flex-1 px-3 py-2 border border-border rounded-lg"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Description (optional)</label>
            <textarea
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              placeholder="A quick call to discuss..."
              rows={2}
              className="w-full px-3 py-2 border border-border rounded-lg resize-none"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Duration</label>
            <select
              value={formData.duration_minutes}
              onChange={(e) => setFormData({ ...formData, duration_minutes: parseInt(e.target.value) })}
              className="w-full px-3 py-2 border border-border rounded-lg"
            >
              <option value={15}>15 minutes</option>
              <option value={30}>30 minutes</option>
              <option value={45}>45 minutes</option>
              <option value={60}>60 minutes</option>
              <option value={90}>90 minutes</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Available Days</label>
            <div className="flex gap-2">
              {dayNames.map((name, index) => (
                <button
                  key={index}
                  type="button"
                  onClick={() => toggleDay(index)}
                  className={`w-10 h-10 rounded-lg text-sm font-medium ${
                    formData.available_days.includes(index)
                      ? 'bg-primary text-white'
                      : 'bg-muted/30 text-muted'
                  }`}
                >
                  {name[0]}
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">Start Time</label>
              <select
                value={formData.start_hour}
                onChange={(e) => setFormData({ ...formData, start_hour: parseInt(e.target.value) })}
                className="w-full px-3 py-2 border border-border rounded-lg"
              >
                {Array.from({ length: 24 }, (_, i) => (
                  <option key={i} value={i}>{formatTime(i)}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">End Time</label>
              <select
                value={formData.end_hour}
                onChange={(e) => setFormData({ ...formData, end_hour: parseInt(e.target.value) })}
                className="w-full px-3 py-2 border border-border rounded-lg"
              >
                {Array.from({ length: 24 }, (_, i) => (
                  <option key={i} value={i}>{formatTime(i)}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">Buffer Before</label>
              <select
                value={formData.buffer_before}
                onChange={(e) => setFormData({ ...formData, buffer_before: parseInt(e.target.value) })}
                className="w-full px-3 py-2 border border-border rounded-lg"
              >
                <option value={0}>No buffer</option>
                <option value={5}>5 minutes</option>
                <option value={10}>10 minutes</option>
                <option value={15}>15 minutes</option>
                <option value={30}>30 minutes</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Buffer After</label>
              <select
                value={formData.buffer_after}
                onChange={(e) => setFormData({ ...formData, buffer_after: parseInt(e.target.value) })}
                className="w-full px-3 py-2 border border-border rounded-lg"
              >
                <option value={0}>No buffer</option>
                <option value={5}>5 minutes</option>
                <option value={10}>10 minutes</option>
                <option value={15}>15 minutes</option>
                <option value={30}>30 minutes</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">Minimum Notice</label>
              <select
                value={formData.advance_notice_hours}
                onChange={(e) => setFormData({ ...formData, advance_notice_hours: parseInt(e.target.value) })}
                className="w-full px-3 py-2 border border-border rounded-lg"
              >
                <option value={1}>1 hour</option>
                <option value={4}>4 hours</option>
                <option value={24}>1 day</option>
                <option value={48}>2 days</option>
                <option value={168}>1 week</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Max Days Ahead</label>
              <select
                value={formData.max_days_ahead}
                onChange={(e) => setFormData({ ...formData, max_days_ahead: parseInt(e.target.value) })}
                className="w-full px-3 py-2 border border-border rounded-lg"
              >
                <option value={7}>1 week</option>
                <option value={14}>2 weeks</option>
                <option value={30}>1 month</option>
                <option value={60}>2 months</option>
                <option value={90}>3 months</option>
              </select>
            </div>
          </div>
        </div>

        <div className="p-4 border-t border-border flex justify-end gap-2">
          <button
            onClick={onClose}
            className="px-4 py-2 border border-border rounded-lg hover:bg-muted/30"
          >
            Cancel
          </button>
          <button
            onClick={() => onSave(formData)}
            disabled={!formData.title || !formData.slug}
            className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary/90 disabled:opacity-50"
          >
            {link ? 'Save Changes' : 'Create Link'}
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * Public booking page component for scheduling links
 */
export function PublicBookingPage({ slug }: { slug: string }) {
  const [link, setLink] = useState<SchedulingLink | null>(null);
  const [selectedDate, setSelectedDate] = useState<Date>(new Date());
  const [availableSlots, setAvailableSlots] = useState<AvailableSlot[]>([]);
  const [booking, setBooking] = useState<BookingFormData>({
    name: '',
    email: '',
    notes: '',
    selectedSlot: null,
  });
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    fetchLinkDetails();
  }, [slug]);

  useEffect(() => {
    if (link) {
      fetchAvailability();
    }
  }, [link, selectedDate]);

  async function fetchLinkDetails() {
    try {
      const response = await fetch(`/api/scheduling/public/${slug}`);
      if (response.ok) {
        setLink(await response.json());
      }
    } catch (error) {
      console.error('Failed to fetch link details:', error);
    } finally {
      setLoading(false);
    }
  }

  async function fetchAvailability() {
    try {
      const response = await fetch(
        `/api/scheduling/public/${slug}/availability?date=${selectedDate.toISOString().split('T')[0]}`
      );
      if (response.ok) {
        setAvailableSlots(await response.json());
      }
    } catch (error) {
      console.error('Failed to fetch availability:', error);
    }
  }

  async function submitBooking() {
    if (!booking.selectedSlot || !booking.name || !booking.email) return;

    setSubmitting(true);
    try {
      const response = await fetch(`/api/scheduling/public/${slug}/book`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start_time: booking.selectedSlot.start,
          attendee_name: booking.name,
          attendee_email: booking.email,
          notes: booking.notes,
        }),
      });
      if (response.ok) {
        setSuccess(true);
      }
    } catch (error) {
      console.error('Failed to book:', error);
    } finally {
      setSubmitting(false);
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (!link) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-2">Link Not Found</h1>
          <p className="text-muted">This scheduling link doesn't exist or has been disabled.</p>
        </div>
      </div>
    );
  }

  if (success) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center max-w-md">
          <div className="text-5xl mb-4">✅</div>
          <h1 className="text-2xl font-bold mb-2">Meeting Booked!</h1>
          <p className="text-muted">
            You'll receive a calendar invite at {booking.email} with the meeting details.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-2xl mx-auto">
        <div className="bg-surface rounded-lg border border-border p-6">
          <h1 className="text-2xl font-bold mb-2">{link.title}</h1>
          {link.description && <p className="text-muted mb-4">{link.description}</p>}
          <div className="flex items-center gap-4 text-sm text-muted mb-6">
            <span>⏱️ {link.duration_minutes} minutes</span>
          </div>

          {!booking.selectedSlot ? (
            <>
              <div className="mb-4">
                <label className="block text-sm font-medium mb-2">Select a Date</label>
                <input
                  type="date"
                  value={selectedDate.toISOString().split('T')[0]}
                  onChange={(e) => setSelectedDate(new Date(e.target.value))}
                  min={new Date().toISOString().split('T')[0]}
                  className="px-3 py-2 border border-border rounded-lg"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Available Times</label>
                {availableSlots.length === 0 ? (
                  <p className="text-muted text-center py-4">No available times on this date</p>
                ) : (
                  <div className="grid grid-cols-3 gap-2">
                    {availableSlots.map((slot, index) => (
                      <button
                        key={index}
                        onClick={() => setBooking({ ...booking, selectedSlot: slot })}
                        className="px-3 py-2 border border-border rounded-lg hover:bg-primary/10 hover:border-primary text-sm"
                      >
                        {new Date(slot.start).toLocaleTimeString([], {
                          hour: '2-digit',
                          minute: '2-digit',
                        })}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="space-y-4">
              <div className="p-4 bg-primary/10 rounded-lg">
                <p className="font-medium">
                  {new Date(booking.selectedSlot.start).toLocaleDateString(undefined, {
                    weekday: 'long',
                    month: 'long',
                    day: 'numeric',
                  })}
                </p>
                <p className="text-sm">
                  {new Date(booking.selectedSlot.start).toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                  {' - '}
                  {new Date(booking.selectedSlot.end).toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </p>
                <button
                  onClick={() => setBooking({ ...booking, selectedSlot: null })}
                  className="text-sm text-primary mt-2"
                >
                  Change time
                </button>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Your Name</label>
                <input
                  type="text"
                  value={booking.name}
                  onChange={(e) => setBooking({ ...booking, name: e.target.value })}
                  className="w-full px-3 py-2 border border-border rounded-lg"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Your Email</label>
                <input
                  type="email"
                  value={booking.email}
                  onChange={(e) => setBooking({ ...booking, email: e.target.value })}
                  className="w-full px-3 py-2 border border-border rounded-lg"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Notes (optional)</label>
                <textarea
                  value={booking.notes}
                  onChange={(e) => setBooking({ ...booking, notes: e.target.value })}
                  rows={3}
                  className="w-full px-3 py-2 border border-border rounded-lg resize-none"
                />
              </div>

              <button
                onClick={submitBooking}
                disabled={submitting || !booking.name || !booking.email}
                className="w-full px-4 py-3 bg-primary text-white rounded-lg hover:bg-primary/90 disabled:opacity-50"
              >
                {submitting ? 'Booking...' : 'Confirm Booking'}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
