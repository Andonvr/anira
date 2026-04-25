import { type AniraWasmInstance } from './factory'
import { BaseWrapper } from './wrappers'

type DropFirst<T extends unknown[]> = T extends [any, ...infer Rest] ? Rest : never

/** A factory function with the wasmInstance pre-bound */
export type Factory<
  C extends new (wasmInstance: AniraWasmInstance, ...args: any[]) => BaseWrapper,
> = ((...args: DropFirst<ConstructorParameters<C>>) => InstanceType<C>) & {
  fromPointer(ptr: number): InstanceType<C>
}

export const createFactory = <
  C extends new (wasmInstance: AniraWasmInstance, ...args: any[]) => BaseWrapper,
>(
  wasmInstance: AniraWasmInstance,
  Cls: C
): Factory<C> => {
  const factory = (...args: DropFirst<ConstructorParameters<C>>) =>
    new Cls(wasmInstance, ...args) as InstanceType<C>

  factory.fromPointer = (ptr: number): InstanceType<C> => {
    const instance = Object.create(Cls.prototype)
    instance.wasmInstance = wasmInstance
    instance.ptr = ptr
    return instance
  }

  return factory as Factory<C>
}
